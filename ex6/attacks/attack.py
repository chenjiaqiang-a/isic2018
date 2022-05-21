
class Attack(object):
    r"""
    Base class for adversarial attacks.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

       
    def forward(self, images, labels):
        r"""
        Defines the computation performed at every call.
        Should be overridden by all subclasses.

        Shape:
            - images: [N, C, H, W] where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: [N] where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        
        Returns:
            adversarial examples: [N, C, H, W].
        """
        raise NotImplementedError


    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack', 'device']

        # for key in info.keys():
        #     if key[0] == "_":
        #         del_keys.append(key)

        for key in del_keys:
            del info[key]

        # info['attack_mode'] = self._attack_mode
        # info['return_type'] = self._return_type

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)

        return images
