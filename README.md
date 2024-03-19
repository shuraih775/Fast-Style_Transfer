# Texture Networks Implementation

This repository contains a custom implementation of the paper titled "Texture Networks: Feed-forward Synthesis of Textures and Stylized Images".

## About the Paper

The work presents an alternative approach to the texture synthesis and stylization problem originally addressed by Gatys et al. Instead of relying on slow and memory-intensive optimization processes, the proposed method shifts the computational burden to a learning stage. By training compact feed-forward convolutional networks, it enables the generation of multiple texture samples from a single example, along with the ability to transfer artistic style from one image to another. The approach yields remarkably lightweight networks capable of generating textures of comparable quality to Gatys et al., but at speeds hundreds of times faster. The implementation demonstrates the effectiveness of generative feed-forward models trained with complex loss functions, achieving synthesis and stylization capabilities comparable to descriptive networks. Additionally, the method introduces a novel multi-scale generative architecture, termed texture networks, which are highly efficient and suitable for various applications, including video-related and potentially mobile scenarios. The paper provides a comprehensive overview of related approaches, describes the methodology, and presents qualitative comparisons demonstrating the efficacy of the approach across challenging textures and images.

## Purpose of the Repository
The purpose of this repository is to provide a custom implementation of the techniques described in the paper. By implementing the methods proposed in the paper, users can explore and experiment with texture synthesis and stylization using deep learning techniques.

## Example
| Content | Style | Result |
| ------- | ----- | ------ |
| ![Content](/example_content.jpg) | ![Style](/example_style.jpg) | ![Result](/example_result.jpg) |


## Contents
- Implementation of texture synthesis and stylization algorithm
- Code for training texture generation model


## Citation
If you use this code for your research or project, please consider citing the original paper:
[Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](https://arxiv.org/pdf/1603.03417v1.pdf)


