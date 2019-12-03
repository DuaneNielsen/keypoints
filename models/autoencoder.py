from knn import Container


class VGGAutoEncoder(Container):
    def __init__(self, encoder, decoder, init_weights=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return z, x