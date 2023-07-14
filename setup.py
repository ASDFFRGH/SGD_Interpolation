from dataclasses import dataclass, asdict

@dataclass
class setup():
    epochs: int = 300
    train_size: int = 10000
    test_size: int = 10000
    data_dimention: int = 10
    noise_std: float = 0.0
    eta: float = 0.1
    momentum: float = 0.0
    weight_decay: float = 0.0
    model_type: str = 'mf'
    mid_layer_size: int = 2**11
    batch_size: int = 10

if __name__ == '__main__':
    s = setup()
    s = asdict(s)
    for key, val in s.items():
        print(f'{key}: {val}')
