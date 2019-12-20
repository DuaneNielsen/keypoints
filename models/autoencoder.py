from models.knn import Container, Identity


class AutoEncoder(Container):
    def __init__(self, name, encoder, decoder,
                 init_weights=True):
        super().__init__(name)
        self.encoder = encoder
        self.decoder = decoder
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return z, x

    def load(self, run_id, epoch):
        self.encoder.load(self.name, run_id, epoch)
        self.decoder.load(self.name, run_id, epoch)

    def save(self, run_id, epoch):
        self.encoder.save(self.name, run_id, epoch)
        self.decoder.save(self.name, run_id, epoch)


