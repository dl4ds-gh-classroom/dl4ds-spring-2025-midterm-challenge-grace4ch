# DS542 Midterm Challenge Report
# Grace Chong


## 1. AI Disclosure Statement

For this project, I used ChatGPT-4 as a supportive resource to deepen my understanding and guide my process. I used it to help me structure PyTorch training and evaluation pipelines, debug shell commands, and set up the SCC environment. I also needed help with understanding complexities and clarifications on the best ways to improve model performance through tuning and regularization. In all, I used AI assistance as a learning tool, similar to a personal tutor, to help reinforce my understanding and support experimentation. In addition, I also asked it to help me reformat my report in order to make it easier to read and look nicer.

## 2. Model Description

### Part 1 – Simple CNN

In Part 1, I implemented a custom CNN architecture from scratch using PyTorch. The model included four convolutional layers with Batch Normalization, ReLU activations, MaxPooling, and Global Average Pooling. A Dropout layer with `p=0.4` and two fully connected layers completed the classifier head. The goal was to establish a working pipeline for training on CIFAR-100, validating on a held-out set, and evaluating on the OOD dataset. This part was also used to integrate Weights & Biases for experiment tracking.

### Part 2 – Manually Implemented ResNet18

In Part 2, I built a ResNet18-style architecture using residual connections implemented via custom `BasicBlock` modules. These identity shortcuts helped improve gradient flow and model depth without vanishing gradients. I experimented with enhancements to improve generalization, including:
- **CosineAnnealingWarmRestarts** as a dynamic learning rate scheduler  
- **Mixup augmentation** to encourage smoother decision boundaries  
- **ColorJitter** and normalization to improve robustness to visual variance  

This deeper network outperformed the Part 1 baseline and generalized better on both clean and OOD datasets.

### Part 3 – Transfer Learning with Pretrained ResNet50

For Part 3, I used a pretrained ResNet50 from the `timm` library and fine-tuned it for CIFAR-100. I replaced the classifier head with a Dropout layer and a `Linear(2048, 100)` output layer. Early experiments froze most of the backbone and progressively unfroze layers (layer4 at epoch 5, layer3 at epoch 10). Later, I experimented with fully unfreezing from the start.

I also tuned:
- **Discriminative learning rates** (lower LR for pretrained layers, higher for new head)  
- **AdamW optimizer** with weight decay  
- **RandAugment**, **Random Erasing**, and **Label Smoothing**  

These enhancements led to the best performance on both validation and OOD sets.  
**Best Kaggle private leaderboard score in Part 3:** `0.48542`

## 3. Hyperparameter Tuning

I experimented with tuning key hyperparameters across all parts, including:
- Learning rates (`0.1`, `5e-4`)
- Weight decay (`5e-4`)
- Batch size (ranged from `8` in Part 1 to `128` in later parts)
- Label smoothing (`0.05`, `0.1`)
- Mixup alpha (`0.2`)
- Optimizer choices (SGD, AdamW)
- Schedulers (CosineAnnealing, CosineAnnealingWarmRestarts)

In Part 3, I explored unfreezing strategies and used discriminative learning rates for different model components.

## 4. Regularization Techniques

To improve generalization and reduce overfitting, I employed several regularization methods:
- **Dropout** in the classifier heads (p = 0.4)
- **Label Smoothing** in Parts 2 and 3
- **Mixup** augmentation in Parts 2 and 3
- **Weight Decay** applied in all training stages

These techniques helped balance underfitting and overfitting, especially in deeper and transfer learning models.

## 5. Data Augmentation Strategy

All models used standard augmentations like:
- **RandomCrop**
- **HorizontalFlip**
- **Normalization**

Advanced augmentations were added in later parts:
- **RandAugment** in Part 3  
- **RandomErasing** in Parts 2 and 3  
- **ColorJitter** in Part 2  

These strategies improved robustness and helped performance on out-of-distribution test samples.

## 6. Results Analysis

Across the three parts, model performance improved progressively:
- **Part 1 (Simple CNN)** served as a functional baseline.  
- **Part 2 (ResNet18)** improved depth and used mixup and color jitter for better accuracy.  
- **Part 3 (ResNet50 Transfer Learning)** yielded the best results with advanced tuning and augmentation.

**Best score (Kaggle private leaderboard):** `0.48542`  
All experiments were tracked using **[Weights & Biases](https://wandb.ai/grace4ch-boston-university/sp25-ds542-challenge)**.  
Evaluation included both clean CIFAR-100 test accuracy and OOD test robustness, verified using `eval_cifar100` and `eval_ood` scripts.

## 7. Conclusion

This project followed a clear learning trajectory—starting from a simple baseline, progressing through a self-built ResNet18, and finally leveraging a pretrained ResNet50 via transfer learning. Each iteration added complexity and refinement in training, tuning, and augmentation.  
Weights & Biases helped track experiments and optimize models quickly. My final model demonstrated strong generalization and surpassed the course benchmark, particularly on OOD evaluation, validating the effectiveness of my chosen strategies.
