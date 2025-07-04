import copy

import torch
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer, RECL_MIX
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
import numpy as np
from utils.th_utils import get_parameters_num

from modules.tactic_selectors import tactic_selector as tactic_selector


class TLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
            self.tactic_mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
            self.tactic_mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.tactic_mixer = Mixer(args).to(args.device)
            self.mixer = RECL_MIX(args).to(args.device)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer).to(args.device)
        self.tactic_target_mixer = copy.deepcopy(self.tactic_mixer).to(args.device)
        self.params += list(self.mixer.parameters())
        self.params += list(self.tactic_mixer.parameters())

        # -----params change
        self.params_drop = list(mac.tactic_selector_drop.parameters())
        aaby_id = {id(param) for param in self.params_drop}
        self.params = [param for param in self.params if id(param) not in aaby_id]

        #
        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.params_v2 = self.params.copy()
        aab = list(mac.tactic_selector.commander.parameters())
        aab_id = {id(param) for param in aab}
        self.params_v2 = [param for param in self.params_v2 if id(param) not in aab_id]
        aac = list(mac.tactic_selector.linear.parameters())
        aac_id = {id(param) for param in aac}
        self.params_v2 = [param for param in self.params_v2 if id(param) not in aac_id]

        # self.params_v2.remove(aaa)
        self.params_agent = list(mac.agent.parameters())
        self.params_agent_id = {id(param) for param in self.params_agent}
        self.params_v2 = [param for param in self.params_v2 if id(param) not in self.params_agent_id]
        self.optimiser_v2 = Adam(params=self.params_v2, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))

        self.optimiser_drop = Adam(params=self.params_drop, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        self.state_dim = args.state_shape
        self.n_agents = args.n_agents

        self.n_actions = args.n_actions
        self.obs_dim = int(np.prod(args.obs_shape))
        self.drop_num = args.drop_num

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

    def tactic_dis_forward(self, batch, t, tactic_outs, mha_out):
        drop_num = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        drop_tactics = []
        drop_comm = []
        for p in drop_num:
            drop_tactic, drop_mha = self.mac.forward(batch, t=t, test_mode=True, drop_prob=p, train_drop=True)
            drop_tactics.append(drop_tactic)
            drop_comm.append(drop_mha)

        loss, loss_comm, loss_tactic = 0, 0, 0
        for i in drop_tactics:
            loss += F.mse_loss(i, tactic_outs)
        for j in drop_comm:
            loss_comm += F.mse_loss(j, mha_out)

        loss = loss/len(drop_tactics)
        loss_comm = loss_comm/len(drop_comm)

        return loss, loss_comm

    def diversity_loss(self, role_embedding):
        batchsize, n, d = role_embedding.shape

        normed_E = F.normalize(role_embedding, p=2, dim=-1)  # (batchsize, n, d)
        sim_matrix = torch.bmm(normed_E, normed_E.transpose(1, 2))  # (batchsize, n, n)

        mask = ~torch.eye(n, dtype=torch.bool, device=role_embedding.device)  # (n, n)
        mask = mask.unsqueeze(0).expand(batchsize, -1, -1)  # (batchsize, n, n)

        diversity_penalty = sim_matrix[mask].reshape(batchsize, -1).mean(dim=1)  # (batchsize,)

        return diversity_penalty.mean()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        agent_tactic_id = batch["agent_tactic_id"][:, :-1]
        # Calculate estimated Q-Values
        self.mac.agent.train()
        self.mac.tactic_selector.train()
        mac_out = []
        # tactic_out = []

        moco_loss = th.tensor(0.0, requires_grad=True).to(self.args.device)
        comm_loss = th.tensor(0.0, requires_grad=True).to(self.args.device)
        tac_loss = th.tensor(0.0, requires_grad=True).to(self.args.device)

        self.mac.init_hidden(batch.batch_size)

        self.mixer.state_gru_hidden = None
        fc_batch_s = F.relu(self.mixer.state_fc(batch["state"].reshape(-1, self.state_dim))).reshape(-1, batch.max_seq_length, self.state_dim)  # shape(batch*max_len+1, state_dim)
        state_gru_outs = []
        role_embeddings = []

        for t in range(batch.max_seq_length):
            agent_outs, _, tactic_outs, role_embedding, mha_out, tactic_em = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            # tactic_out.append(tactic_outs)
            role_embeddings.append(role_embedding)

            moco_loss_1, comm_loss_1 = self.tactic_dis_forward(batch, t, tactic_outs.clone().detach(), mha_out.clone().detach())
            tac_loss_1 = self.diversity_loss(tactic_em)
            moco_loss += moco_loss_1
            comm_loss += comm_loss_1
            tac_loss += tac_loss_1

            self.mixer.state_gru_hidden = self.mixer.state_gru(fc_batch_s[:, t].reshape(-1, self.state_dim), self.mixer.state_gru_hidden)  # shape=(batch, N*state_embed_dim)
            state_gru_outs.append(self.mixer.state_gru_hidden)

        moco_loss /= (batch.batch_size * batch.max_seq_length * self.n_agents)
        comm_loss /= (batch.batch_size * batch.max_seq_length * self.n_agents)
        tac_loss /= (batch.batch_size * batch.max_seq_length * self.n_agents)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        role_embeddings = th.stack(role_embeddings, dim=1)

        state_gru_outs = th.stack(state_gru_outs, dim=1).reshape(-1, self.n_agents, self.args.state_embed_dim)  # shape=(batch*max_len+1, N,state_embed_dim)

        role_embeddings = role_embeddings.reshape(-1, self.n_agents, self.args.input_shape)
        att_eval = self.mixer.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, batch.max_seq_length, self.n_agents * self.args.att_out_dim)  # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)
        with th.no_grad():
            att_target = self.target_mixer.attention_net(
                state_gru_outs, role_embeddings, role_embeddings).reshape(-1, batch.max_seq_length, self.n_agents * self.args.att_out_dim)  # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            self.target_mac.tactic_selector.train()
            target_mac_out = []
            # target_tactic_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, _, target_tactic_outs, _, _, _ = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                # target_tactic_out.append(target_tactic_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
            # target_tactic_out = th.stack(target_tactic_out, dim=1)
            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            # cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            cur_max_actions = th.argmax(mac_out_detach, dim=3, keepdim=True)
            target_max_qvals = th.gather(target_mac_out[:, 1:], 3, cur_max_actions[:, 1:]).squeeze(3)

        # ----new loss
        q_total_eval = self.mixer(chosen_action_qvals, fc_batch_s[:, :-1], att_eval[:, :-1])
        q_total_target = self.target_mixer(target_max_qvals, fc_batch_s[:, 1:], att_target[:, 1:])
        targets = batch["reward"][:, :-1] + self.args.gamma * (1 - terminated) * q_total_target
        td_error = (q_total_eval - targets.detach())  # targets.detach() to cut the backward
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / (1 - terminated).sum()

        loss = loss + tac_loss

        sum_loss = comm_loss + 0.5 * moco_loss
        # sum_loss = loss + moco_loss

        last_log_T, change_label = 0, 0
        if (t_env - last_log_T) >= self.args.log_interval:
            change_label += 1

        remainder = change_label % 20
        if 0 <= remainder < 10:
            self.optimizer = self.optimiser
            self.parameters = self.params
        else:
            self.optimizer = self.optimiser_v2
            self.parameters = self.params_v2

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        self.optimiser_drop.zero_grad()
        sum_loss.backward()
        grad_norm_drop = th.nn.utils.clip_grad_norm_(self.params_drop, self.args.grad_norm_clip)
        self.optimiser_drop.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("moco_loss", moco_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                        / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                         / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.tactic_target_mixer.load_state_dict(self.tactic_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()

        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.tactic_mixer.cuda()
            self.tactic_target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))  # tactic完的模型
            th.save(self.tactic_mixer.state_dict(), "{}/tactic_mixer.th".format(path))
            # th.save(self.tactic_prob.state_dict(), "{}/tactic_prob.th".format(path))  # 存drop的前一半模型
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.tactic_mixer.load_state_dict(
                th.load("{}/tactic_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
