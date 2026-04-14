import torch
import transformer_flow


class BridgeTokenizer(torch.nn.Module):
    def __init__(
        self,
        flow_dim: int,
        num_patches: int,
        core_dim: int = 64,
        depth: int = 4,
        head_dim: int = 32,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.flow_dim = flow_dim
        self.core_dim = core_dim

        self.in_proj = torch.nn.Linear(flow_dim, core_dim)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, core_dim) * 1e-2)

        self.blocks = torch.nn.ModuleList(
            [
                transformer_flow.AttentionBlock(
                    channels=core_dim,
                    head_channels=head_dim,
                    expansion=4,
                )
                for _ in range(depth)
            ]
        )
        self.norm = torch.nn.LayerNorm(core_dim)
        self.out_proj = torch.nn.Linear(core_dim, flow_dim)

    def to_core(self, zf):
        x = self.in_proj(zf) + self.pos_embed[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def to_flow(self, zs):
        return self.out_proj(zs)

    def forward(self, zf):
        zs = self.to_core(zf)
        zf_hat = self.to_flow(zs)
        return zs, zf_hat