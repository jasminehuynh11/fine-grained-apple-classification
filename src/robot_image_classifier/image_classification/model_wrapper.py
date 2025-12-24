import torch
from torchvision import models

class EfficientNetB0GroupFineTuner:
    """
    Wrapper class for loading and using a fine-tuned EfficientNet-B0 model
    for fine-grained fruit classification.
    
    The model was initially trained on 21 fruit classes (8 apples, 2 oranges,
    5 grapes, 2 bananas, 4 pears), but for deployment purposes, it is fine-tuned
    and deployed for 3 grape classes: Autumn Royal, Crimson Seedless, and Thompson Seedless.
    
    This wrapper handles model loading, device placement, and provides a clean
    interface for inference.
    """

    def __init__(self,
                 checkpoint_path: str,
                 num_classes: int = 3,
                 device: torch.device = None):
        # 1. Decide device
        self.device = (
            device
            or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 2. Build the model architecture
        self.model = models.efficientnet_b0(num_classes=num_classes)

        # 3. Load checkpoint (expects you saved 'model_state_dict', 'class_to_idx', 'class_names')
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()

        # 4. (Optional) restore mapping info if you saved it
        self.class_to_idx = ckpt.get('class_to_idx', None)
        self.class_names  = ckpt.get('class_names', None)
        
        # 5. Get idx_to_class
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def predict(self, input_tensor: torch.Tensor):
        """
        Run a forward pass on a preprocessed batch tensor.
        Returns (confidences_tensor, predicted_index_tensor).
        """
        with torch.no_grad():
            logits = self.model(input_tensor.to(self.device))
            probs  = torch.nn.functional.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
        return conf.cpu(), idx.cpu()
