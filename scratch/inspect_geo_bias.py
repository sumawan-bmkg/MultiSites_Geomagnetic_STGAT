import torch

device = torch.device('cpu')
ckpt_path = 'src/stgat_v2/checkpoints/stgat_resilient_best.pth'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = ckpt.get('model_state_dict', ckpt)

if 'gat_layer.geo_bias' in state_dict:
    geo_bias = state_dict['gat_layer.geo_bias']
    print("=== learned geo_bias matrix ===")
    print(f"Shape: {geo_bias.shape}")
    print(f"Mean absolute value: {torch.mean(torch.abs(geo_bias)).item():.6f}")
    print(f"Min value: {torch.min(geo_bias).item():.6f}")
    print(f"Max value: {torch.max(geo_bias).item():.6f}")
    print("\nFirst 5x5 subgrid:")
    print(geo_bias[:5, :5])
else:
    print("gat_layer.geo_bias not found in state dict!")
    print("Available keys in state_dict:")
    print([k for k in state_dict.keys() if 'gat_layer' in k])
