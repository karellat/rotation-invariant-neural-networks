import time
import torch

import escnn
from escnn import gspaces

from hippy2d.learnable import flusser_basis_orders
from hippy2d.opt_inv_layers import OptimalBasisBlock

ITERS = 50
MAX_ORDER = 4
IN_CHANNELS = 3
HID_CHANNELS = 16
SPATIAL = 64
BATCH_SIZE = 256

def init_models(max_order, device): 
    orders = flusser_basis_orders(max_order)
    r2_act = gspaces.rot2dOnR2(N=-1)
    # Create analogical representation to flusser moments
    feat_in = escnn.nn.FieldType(r2_act, 3 * [r2_act.trivial_repr])

    hid_repr = []

    for p in orders:
        rep = r2_act.irreps[p]
        for _ in range(HID_CHANNELS):
            hid_repr.append(rep)

    feat_hid = escnn.nn.FieldType(r2_act, hid_repr)
    print(f"Hidden representation: {feat_hid}")

    # Create the block
    class _ours(torch.nn.Module):
        def __init__(self, IN_CHANNELS, HID_CHANNELS, orders):
            super().__init__()
            self.block1 = OptimalBasisBlock(in_channels=IN_CHANNELS, 
                                            out_channels=None,
                                            phase_func="polar", 
                                            orders=orders) 
            self.conv1x1 = torch.nn.Conv2d(in_channels=IN_CHANNELS * self.block1.invariants_layer.out_channels,
                                           out_channels=HID_CHANNELS, 
                                           kernel_size=1, 
                                           bias=False)
            self.block2 = OptimalBasisBlock(in_channels=HID_CHANNELS,
                                            out_channels=None,
                                            phase_func="polar", 
                                            orders=orders)
        def forward(self, x):
            x = self.block1(x)
            x = self.conv1x1(x)
            x = self.block2(x)
            return x
    ours = _ours(IN_CHANNELS, HID_CHANNELS, orders).to(device)

    class esnn(torch.nn.Module):
        def __init__(self, feat_in, feat_hid, r2_act):
            super().__init__()
            self.feat_in = feat_in
            self.conv1 = escnn.nn.R2Conv(feat_in, feat_hid, kernel_size=11, padding=5)
            self.conv2 = escnn.nn.R2Conv(feat_hid, feat_hid, kernel_size=11, padding=5)

        def forward(self, x):
            x = self.feat_in(x)
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    esnn = esnn(feat_in, feat_hid, r2_act).to(device)
    return dict(ours=ours, esnn=esnn)


def main():
    # Prepare sample data 
    times = dict(ours=[], esnn=[])
    # Warm up

    for ord in [1, 2, 3, 4, 5, 6]:
        shape = [BATCH_SIZE, IN_CHANNELS, SPATIAL, SPATIAL]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        v = torch.randn(*shape, device=device)
        print(f"Testing order {ord}...")
        models = init_models(ord, device)
        for model_name, model in models.items():
            for _ in range(20):
                _ = model(v)
            if device == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(ITERS):
                _ = model(v)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            total_time = (t1 - t0) / ITERS
            print(model_name, total_time)
            times[model_name].append(total_time)

        del models
        # clear cuda cache
        if device == "cuda":
            torch.cuda.empty_cache()
    # Save as csv
    import pandas as pd
    df = pd.DataFrame(times)
    # Time stamp 
    stamp = time.strftime("%Y%m%d-%H%M%S")
    df.to_csv(f"layer_timing_{stamp}.csv", index=False)




if __name__ == "__main__":
    main()
