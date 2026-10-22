"""Strict checkpoint loading, forward pass, and one-step training checks."""
import argparse
import json
from _common import ROOT, load_model
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--training-step', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    report = {}
    for ds in ['suim', 'dut']:
        model, cfg, meta = load_model(ds, args.device)
        x = torch.randn(2, 3, 64, 96, device=args.device)
        head = model.decode_head
        before = {k: v.clone() for k, v in head.named_buffers()}
        with torch.no_grad():
            logits = head(model.extract_feat(x))
        assert logits.shape[:2] == (2, len(model.CLASSES))
        assert torch.isfinite(logits).all()
        assert all(torch.equal(v, before[k]) for k, v in head.named_buffers())
        entry = dict(parameters=sum(v.numel() for v in model.parameters()),
                     classes=list(model.CLASSES), checkpoint_iteration=meta.get('iter'),
                     strict_load=True, forward_finite=True, inference_buffers_unchanged=True)
        if args.training_step:
            model.train()
            gt = torch.randint(len(model.CLASSES), (2, 1, 64, 96), device=args.device)
            loss_dict = model.forward_train(x, [{}, {}], gt)
            loss = sum(value for key, value in loss_dict.items() if 'loss' in key)
            assert torch.isfinite(loss)
            loss.backward()
            assert head.text_prototypes.grad is not None
            assert head.gate_conv.weight.grad is None, 'Historical detach_gate semantics changed'
            torch.optim.AdamW(model.parameters(), lr=1e-6).step()
            entry.update(training_step_finite=True, embedding_gradient=True, gate_gradient=False)
        report[ds] = entry
        print(ds, json.dumps(entry), flush=True)
    out = ROOT / 'results/preflight.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
