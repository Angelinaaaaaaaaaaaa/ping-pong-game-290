import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Flat-concat MLP: [s, k1, k2] -> BatchNorm -> MLP -> scalar.

    Baseline estimator architecture used by the v2 Nash pipeline for both
    Q-value models (one per player) and the alpha-potential model.
    """

    def __init__(self, input_size, hidden_size, output_size, last_layer_activation='tanh'):
        super(SimpleModel, self).__init__()
        self.fc = []
        self.batch_norm = nn.BatchNorm1d(input_size, affine=False)
        curr_h = input_size
        for h in hidden_size:
            self.fc.append(nn.Linear(curr_h, h))
            self.fc.append(nn.ReLU())
            curr_h = h
        self.fc.append(nn.Linear(curr_h, output_size))
        if last_layer_activation == 'tanh':
            self.fc.append(nn.Tanh())
        elif last_layer_activation == 'sigmoid':
            self.fc.append(nn.Sigmoid())
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.batch_norm(x)
        elif len(x.shape) == 3:
            x1 = self.batch_norm(x.reshape((-1, x.shape[-1])))
            x = x1.reshape(x.shape)
        out = self.fc(x)
        return out


class FactoredModel(nn.Module):
    """Factored architecture for the §3.6 ablation.

    Enc_s(s) and Enc_k(k1,k2) run on the state and skill slices separately,
    their outputs are concatenated and fed into a fusion MLP that produces a
    scalar. Input layout matches SimpleModel: the last `skill_dim` columns of
    the input are the skill encoding; everything before them is the state.

    Default skill_dim=2 covers the (ego_skill, opp_skill) layout used by both
    the 2-skill and 5-skill v2 pipelines. The skill columns are normalised
    indices ([0,1] for 5-skill) or ±1 (for 2-skill); both are 2-dim so the
    same class works in either pipeline.
    """

    def __init__(self, state_dim, skill_dim=2, hidden=None,
                 output_size=1, last_layer_activation='tanh',
                 state_hidden=32, skill_hidden=8):
        super().__init__()
        if hidden is None:
            hidden = [64, 32, 16]
        self.state_dim = state_dim
        self.skill_dim = skill_dim

        # BatchNorm on the full input keeps parity with SimpleModel for a clean
        # estimator-only ablation. affine=False matches SimpleModel.
        self.batch_norm = nn.BatchNorm1d(state_dim + skill_dim, affine=False)

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden), nn.ReLU(),
        )
        self.skill_encoder = nn.Sequential(
            nn.Linear(skill_dim, skill_hidden), nn.ReLU(),
        )

        fusion = []
        curr_h = state_hidden + skill_hidden
        for h in hidden:
            fusion.append(nn.Linear(curr_h, h))
            fusion.append(nn.ReLU())
            curr_h = h
        fusion.append(nn.Linear(curr_h, output_size))
        if last_layer_activation == 'tanh':
            fusion.append(nn.Tanh())
        elif last_layer_activation == 'sigmoid':
            fusion.append(nn.Sigmoid())
        self.fusion = nn.Sequential(*fusion)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.batch_norm(x)
        elif len(x.shape) == 3:
            x = self.batch_norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        s_feat = self.state_encoder(x[..., :self.state_dim])
        k_feat = self.skill_encoder(x[..., self.state_dim:])
        return self.fusion(torch.cat([s_feat, k_feat], dim=-1))
