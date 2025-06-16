from mmcv.runner import HOOKS, Hook
import torch

@HOOKS.register_module()
class FreezeAllButNewDepthHook(Hook):
    def before_train_epoch(self, runner):
        print('[HOOK] FreezeAllButNewDepthHook is running')
        model = runner.model
        model = model.module if hasattr(model, 'module') else runner.model

        for name, param in model.named_parameters():
        # dinov2 в адаптере уже заморожен, надо дополнительно заморозить все, кроме 'dino'
            if 'dino' in name or 'img_feats_compr' in name:
                pass
            else:
                param.requires_grad = False

        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
                m.eval()


@HOOKS.register_module()
class CheckFrozenParamsHook(Hook):

    def __init__(self):
        self.param_snapshots = {}

    def before_train_epoch(self, runner):
        print('[HOOK] CheckFrozenParamsHook is running')
        model = runner.model
        self.param_snapshots = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            # if not param.requires_grad
        }

    def after_train_epoch(self, runner):
        model = runner.model
        print("[CheckFrozenParamsHook] Comparing frozen params...")
        for name, old_param in self.param_snapshots.items():
            current_param = dict(model.named_parameters())[name]
            if not torch.allclose(current_param.detach(), old_param, atol=1e-8):
                print(f"❗ Parameter '{name}' has changed")
        print("[CheckFrozenParamsHook] Done.\n")
