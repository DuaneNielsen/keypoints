import torch
from higgham import isPD, nearestPDHack
from tqdm import tqdm


def test_higgham():

    with torch.no_grad():
        for i in range(10):
            for j in tqdm(range(2, 100)):
                print('#')
                A = torch.randn(j, j)
                B = nearestPDHack(A)
                assert (isPD(B))
        print('unit test passed!')