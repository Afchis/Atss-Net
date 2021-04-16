import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechEmbedder(nn.Module):
    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(40,
                            768,
                            num_layers=3,
                            batch_first=True)
        self.proj = nn.Linear(768, 256)
        self.loss_w = nn.Parameter(torch.tensor([10.]))
        self.loss_b = nn.Parameter(torch.tensor([-5.]))
        self.loss_w.requires_grad = True
        self.loss_b.requires_grad = True
        self.sigmoid = nn.Sigmoid()

    def _neg_centroids(self, dvec):
        return torch.mean(dvec, dim=1, keepdim=True)

    def _pos_centroid(self, dvec, sp_idx, utt_idx):
        pos_cent = torch.cat([dvec[sp_idx, :utt_idx], dvec[sp_idx, utt_idx+1:]], dim=0)
        return torch.mean(pos_cent, dim=0, keepdim=True)

    def _sim_matrix(self, dvec):
        neg_centroids = self._neg_centroids(dvec)
        '''
        dvec.shape          --> [speaker_idx, utterance_idx, emb_dim]
        neg_cintroids.shape --> [speaker_idx, 1, emb_dim]
        pos_centroid.shape --> [1, emb_dim]
        '''
        pos_sim = list()
        neg_sim = list()
        for sp_idx in range(dvec.size(0)):
            pos_sim_speaker = list()
            neg_sim_speaker = list()
            for utt_idx in range(dvec.size(1)):
                # pos sim:
                pos_centroid = self._pos_centroid(dvec, sp_idx, utt_idx)
                pos_sim_utt = self.loss_w * F.cosine_similarity(dvec[sp_idx, utt_idx].reshape(1, -1), pos_centroid, dim=1, eps=1e-6) + self.loss_b # [1]
                pos_sim_speaker.append(pos_sim_utt)
                # neg sim:
                neg_sim_utt = self.loss_w * F.cosine_similarity(dvec[sp_idx, utt_idx].reshape(1, -1), torch.cat([neg_centroids[:sp_idx], neg_centroids[sp_idx+1:]], dim=0).squeeze(), dim=1) + self.loss_b # [speaker_idx-1]
                neg_sim_speaker.append(neg_sim_utt)
            pos_sim_speaker = torch.stack(pos_sim_speaker, dim=0)
            pos_sim.append(pos_sim_speaker)
            neg_sim_speaker = torch.stack(neg_sim_speaker, dim=0)
            neg_sim.append(neg_sim_speaker)
        pos_sim = torch.stack(pos_sim, dim=0) # [speaker_idx, utterance_idx, 1]
        neg_sim = torch.stack(neg_sim, dim=0) # [speaker_idx, utterance_idx, speaker_idx-1]
        return pos_sim, neg_sim
 
    def _contrast_loss(self, pos_sim, neg_sim):
        loss =  1 - self.sigmoid(pos_sim.squeeze()) + self.sigmoid(neg_sim).max(2)[0]
        return loss.mean(), pos_sim.mean().item(), neg_sim.mean().item()

    def _softmax_loss(self, pos_sim, neg_sim):
        loss = - pos_sim.squeeze() + torch.log(torch.exp(neg_sim).sum(dim=2))
        return loss.mean(), pos_sim.mean().item(), neg_sim.mean().item() #, torch.log(torch.exp(neg_sim).sum(dim=2)).mean().item()

    def _ge2e_loss(self, dvec):
        dvec = dvec.reshape(self.speaker_num, self.utterance_num, -1)
        torch.clamp(self.loss_w, 1e-6)
        pos_sim, neg_sim = self._sim_matrix(dvec)
        loss, pos_sim, neg_sim = self._softmax_loss(pos_sim, neg_sim)
        return loss, pos_sim, neg_sim

    def forward(self, mel):
        self.speaker_num, self.utterance_num = mel.size(0), mel.size(1)
        mel = mel.reshape(self.speaker_num*self.utterance_num, mel.size(3), mel.size(4))
        # (b, c, num_mels, T) b --> [speaker_idx*utterance_idx] time = 94
        # (b, num_mels, T)   c == 1
        mel = mel.permute(0, 2, 1) # (b, T, num_mels)
        dvec, _ = self.lstm(mel) # (b, T, lstm_hidden)
        if self.train:
            dvec = dvec[:, -1, :]
            dvec = self.proj(dvec) # (b, emb_dim)
            dvec = dvec / dvec.norm(p=2, dim=1, keepdim=True)
            loss, pos_sim, neg_sim = self._ge2e_loss(dvec)
            return loss, pos_sim, neg_sim
        else:
            dvec = dvec.sum(1) / dvec.size(1)
            dvec = self.proj(dvec) # (b, emb_dim)
            dvec = dvec / dvec.norm(p=2, dim=1, keepdim=True)
            return dvec

