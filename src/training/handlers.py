import os
import torch
import ignite.handlers as hdlr


class ModelCheckpoint(hdlr.ModelCheckpoint):
    @property
    def last_checkpoint_state(self):
        with open(self.last_checkpoint, mode='rb') as f:
            state_dict = torch.load(f)
        return state_dict

    @property
    def all_paths(self):
        def name_path_tuple(p):
            return p.filename, os.path.join(self.save_handler.dirname,
                                            p.filename)

        return [name_path_tuple(p) for p in self._saved]
