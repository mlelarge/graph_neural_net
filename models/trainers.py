import pytorch_lightning as pl
from toolbox.losses import triplet_loss
from toolbox.metrics import accuracy_max, accuracy_linear_assignment
from models.blocks_emb import *
from models.utils import *
#from  toolbox.utils import schedule

get_node_emb = {
    'node_embedding': node_embedding,
    }

get_block_init = {
    'block_emb': block_emb
}
    
get_block_inside = {
    'block': block
    }

class Siamese_Node_Exp(pl.LightningModule):
    def __init__(self, original_features_num, node_emb, lr=1e-3, scheduler_decay=0.5, scheduler_step=3, lr_stop = 1e-5):
        """
        take a batch of pair of graphs as
        (bs, original_features, n_vertices, n_vertices)
        and return a batch of "node similarities (i.e. dot product)"
        with shape (bs, n_vertices, n_vertices)
        graphs must NOT have same size inside the batch when maskedtensors are used
        """
        super().__init__()
        try:
            node_emb_type = get_node_emb[node_emb['type']]
        except KeyError:
            raise NotImplementedError(f"node embedding {node_emb['type']} is not implemented")
        try:
            block_inside = get_block_inside[node_emb['block_inside']]
            node_emb['block_inside'] = block_inside
        except KeyError:
            raise NotImplementedError(f"block inside {node_emb['block_inside']} is not implemented")
        try:
            block_init = get_block_init[node_emb['block_init']]
            node_emb['block_init'] = block_init
        except KeyError:
            raise NotImplementedError(f"block init {node_emb['block_init']} is not implemented")

        self.out_features = node_emb['out_features']
        self.node_embedder_dic = {
            'input': (None, []), 
            'ne':  node_emb_type(original_features_num, **node_emb)
                }
        self.node_embedder = Network(self.node_embedder_dic)
        
        self.loss = triplet_loss()
        self.metric = accuracy_linear_assignment#accuracy_max
        self.lr = lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.lr_stop = lr_stop
        

    def forward(self, x1, x2):
        """
        Data should be given with the shape (b,2,f,n,n)
        """
        x1 = self.node_embedder(x1)['ne/suffix']
        x2 = self.node_embedder(x2)['ne/suffix']
        #raw_scores = torch.einsum('bfi,bfj-> bij', x1, x2)
        raw_scores = torch.matmul(torch.transpose(x1,1,2),x2)
        return raw_scores
    
    def training_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores)
        self.log('train_loss', loss)
        (acc,n) = self.metric(raw_scores)
        self.log("train_acc", acc/n)
        return loss

    def validation_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores)
        self.log('val_loss', loss)
        (acc,n) = self.metric(raw_scores)
        self.log("val_acc", acc/n)

    def test_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores)
        self.log('test_loss', loss)
        (acc,n) = self.metric(raw_scores)
        self.log("test_acc", acc/n)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                            amsgrad=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_decay, patience=self.scheduler_step, verbose=True, min_lr=self.lr_stop),
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }