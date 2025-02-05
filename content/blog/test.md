---
title: Two-Min Papers by Myself
date: 2024-11-11T07:51:22.229Z
---

- **VanGogh: A Unified Multimodal Diffusion-based Framework for Video Colorization**
    - Dual Qformer to align and fuse features from multiple modalities, complemented by a depth-guided generation process and an optical flow loss, which help reduce color overflow
    - a color injection strategy and luma channel replacement are implemented to improve generalization and mitigate flickering artifacts
    
- **VideoAuteur: Towards Long Narrative Video Generation**
    - We build CookGen, a large, structured dataset and a comprehensive data pipeline designed to benchmark longform narrative video generation. We will open-source the data along with the necessary functionalities to support future long video generation research.
    - We propose VideoAuteur, an effective data-driven pipeline for automatic long video generation. We emperically explore the design and training of an interleaved image-text auto-regressive model for generating visual states and a visual-conditioned video generation model.

    
- **VideoRAG: Retrieval-Augmented Generation over Video Corpus**
    - a novel framework that not only dynamically retrieves relevant videos based on their relevance with queries but also utilizes both visual and textual information of videos in the output generation. Further, to operationalize this, our method revolves around the recent advance of Large Video Language Models (LVLMs), which enable the direct processing of video content to represent it for retrieval and seamless integration of the retrieved videos jointly with queries.

    
- **Multimodal LLMs Can Reason about Aesthetics in Zero-Shot**
    - datasets + finetuning: We introduce MM-StyleBench, the first large-scale dataset for multimodal stylization with dense annotations
    - We propose a principled approach for modeling human aesthetic preference
- **Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise**
    - Specifically, we propose a novel noise warping algorithm, fast enough to run in real time, that replaces random temporal Gaussianity with correlated warped noise derived from optical flow fields, while preserving the spatial Gaussianity

    
- **BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations**
    - we develop a blob-grounded video diffusion model named BlobGEN-Vid that allows users to control object motions and fine-grained object appearance
    - In particular, we introduce a masked 3D attention module that effectively improves regional consistency across frames. In addition, we introduce a learnable module to interpolate text embeddings so that users can control semantics in specific frames and obtain smooth object transitions.
    - All training processes are done on 64 or 128 NVIDIA A100 GPUs.

    
- **Compositional Text-to-Image Generation with Dense Blob Representations**
    - Particularly, we introduce a new masked cross-attention module to disentangle the fusion between blob representations and visual features. To leverage the compositionality of large language models (LLMs), we introduce a new in-context learning approach to generate blob representations from text prompts.
- **One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt**
    - Drawing inspiration from the inherent context consistency, we propose a novel training-free method for consistent text-to-image (T2I) generation, termed “One-Prompt-One-Story” (1Prompt1Story).
    - Our approach 1Prompt1Story **concatenates** all prompts into a single input for T2I diffusion models, initially preserving character identities. We then **refine** the generation process using two novel techniques: Singular-Value Reweighting and Identity-Preserving Cross-Attention, ensuring better alignment with the input description for each frame.

- **MangaNinja: Line Art Colorization with Precise Reference Following**
    - a patch shuffling module to facilitate correspondence learning between the reference color image and the target line art,
    - and a pointdriven control scheme to enable fine-grained color matching.

    
- **Training-Free Motion-Guided Video Generation with Enhanced Temporal Consistency Using Motion Consistency Loss**
    - We design an approach that represents the reference motion through the inter-frame feature correlation patterns of sparse points and transfers motion by replicating these reference matching patterns

- **Learnings from Scaling Visual Tokenizers for Reconstruction and Generation**
    - We first study how scaling the auto-encoder bottleneck affects both reconstruction and generation – and find that while it is highly correlated with reconstruction, its relationship with generation is more complex.
    - Crucially, we find that scaling the encoder yields minimal gains for either reconstruction or generation, while scaling the decoder boosts reconstruction but the benefits for generation are mixed.

- **Can We Generate Images with CoT? Let’s Verify and Reinforce Image Generation Step by Step**
    - scaling test-time computation for verification, aligning model preferences with Direct Preference Optimization (DPO), and integrating these techniques for complementary effects. Our results demonstrate that these approaches can be effectively adapted and combined to significantly improve image generation performance.
    - Furthermore, given the pivotal role of reward models in our findings, we propose the Potential Assessment Reward Model (PARM) and PARM++, specialized for autoregressive image generation. PARM adaptively assesses each generation step through a potential assessment approach, merging the strengths of existing reward models, and PARM++ further introduces a reflection mechanism to self-correct the generated unsatisfactory image.

- **A General Framework for Inference-time Scaling and Steering of Diffusion Models**
    - Our results demonstrate that inference-time scaling and steering of diffusion models – even with off-the-shelf rewards – can provide significant sample quality gains and controllability benefits.
    - FK STEERING enables guidance with arbitrary reward functions, differentiable or otherwise, for both discrete and continuous-state space models.
    - FK STEERING works by (a) sampling multiple interacting diffusion processes, called particles, (b) scoring these particles using functions called potentials, and (c) resampling the particles based on their potentials at intermediate steps during generation

- **Yi: Open Foundation Models by [01.AI](http://01.ai/)**
    - data scaling + pre-training + SFT + long-context tuning + VLM (vision encoder)
    - LLAMA architecture
    
    
- **DeepSeek-VL: Towards Real-World Vision-Language Understanding**
    - Data: We strive to ensure our data is diverse, scalable and extensively covers real-world scenarios including web screenshots, PDFs, OCR, charts, and knowledge-based content (expert knowledge, textbooks), 以及一个高质量的fine-tuning datasets；
    - Architecture：DeepSeek-VL incorporates a hybrid vision encoder that efficiently processes high-resolution images (1024 x 1024) within a fixed token budget, while maintaining a relatively low computational overhead。 As such, we employ a hybrid vision encoder, which combines a text-aligned encoder for coarse semantic extraction at 384 × 384 resolution with a high-resolution encoder that captures detailed visual information at 1024 × 1024 resolution.
        
        
    - Training：To ensure the preservation of LLM capabilities during pretraining, we investigate an effective VL pretraining strategy by integrating LLM training from the beginning and carefully managing the competitive dynamics observed between vision and language modalities
- **Rectified Diffusion Guidance for Conditional Generation**
    - In this work, we revisit the theory behind CFG and rigorously confirm that the improper configuration of the combination coefficients (i.e., the widely used summing-to-one version) brings about expectation shift of the generative distribution. To rectify this issue, we propose ReCFG1 with a relaxation on the guidance coefficients such that denoising with ReCFG strictly aligns with the diffusion theory
    - We further show that our approach enjoys a closed-form solution given the guidance strength. That way, the rectified coefficients can be readily pre-computed via traversing the observed data, leaving the sampling speed barely affected.
- **Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling**
    - (1) an optimized training strategy, (2) expanded training data, and (3) scaling to larger model size.
    - Longer Training in Stage I: We increase the training steps in Stage I, allowing sufficient training on the ImageNet dataset. Our findings reveals that even with the LLM parameters fixed, the model could effectively model pixel dependence and generate reasonable images based on category names. • Focused Training in Stage II: In Stage II, we drop ImageNet data and directly utilize normal text-to-image data to train the model to generate images based on dense descriptions. This redesigned approach enables Stage II to utilize the text-to-image data more efficiently, resulting in improved training efficiency and overall performance
    - In Janus-Pro, we scaled the model up to 7B, We observe that when utilizing a larger-scale LLM, the convergence speed of losses for both multimodal understanding and visual generation improved significantly compared to the smaller model.
    
    
- **RelightVid: Temporal-Consistent Diffusion Model for Video Relighting**
    - video domain的relighting，整体思路很像animate-diff，在IC-light的基础上支持temporal layers，然后收集一波in-the-wild的数据集进行训练；
    - We adopt the SD-1.5 [Rombach et al. 2022] version of IC-Light [Zhang et al. 2024] as the image backbone and inject temporal attention layers initialized from AnimateDiff-V2 [Guo et al. 2023].
    
    
- **CascadeV: An Implementation of Wurstchen Architecture for Video Generation**
    - The base T2V model generates latnet representations aligned with textual semantic information, which serves as a conditional input for LDM-VAE. The VAE decoder, comprising a latent diffusion model and a standard VAE, augments the base model’s output with high-frequency details, achieving a 32× decoding ratio.
    
    
- **SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer**
    - (1) Efficient Training Scaling: A depth-growth paradigm that enables scaling from 1.6B to 4.8B parameters with significantly reduced computational resources, combined with a memory-efficient 8-bit optimizer，使用小模型来初始化大模型的权重，而不是从头训练；
    - (2) Model Depth Pruning: A block importance analysis technique for efficient model compression to arbitrary sizes with minimal quality loss，根据importance （we analyze block importance through input-output similarity patterns）来选择；
    - (3) Inference-time Scaling: A repeated sampling strategy that trades computation for model capacity, enabling smaller models to match larger model quality at inference time. （ scaling the number of sampling， NVILA-2B [14] and developed a specialized dataset to fine-tune it for evaluating images）
    
    
    - The **model growth strategy** first explores a larger optimization space, discovering better feature representations. The **model depth pruning** then identifies and preserves these essential features, enabling efficient deployment. Meanwhile, **inference-time scaling** provides a complementary perspective. When model capacity is constrained, we can utilize extra inference-time computational resources to achieve similar or even better results than larger models.
    
- **Diffusion Autoencoders are Scalable Image Tokenizers**
    - Our key insight is that a single learning objective, diffusion L2 loss, can be used for training scalable image tokenizers.
    - Our results show that DiTo is a simpler, scalable, and self-supervised alternative to the current state-of-the-art image tokenizer which is supervised.
    - At inference, given the latent z, the decoder reconstructs the image from the latent with a diffusion sampler
    - We also find that image generation models trained on DiTo latent representations are competitive to or outperform those trained on GLPTo representations. DiTo can easily be scaled up by increasing the size of the model without requiring any further tuning of loss hyperparameters
    
    
- **Cosmos World Foundation Model Platform for Physical AI**
    - Data curation + tokenizer + diffusion/AR
    - We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers.
    - **Data determines the ceiling of an AI model.** To build a high-ceiling pre-trained WFM, we develop a video data curation pipeline. We use it to locate portions of videos with rich dynamics and high visual quality that facilitate learning of physics encoded in visual content. We use the pipeline to extract about 100M clips of videos ranging from 2 to 60 seconds from a 20M hour-long video collection. For each clip, we use a visual language model (VLM) to provide a video caption per 256 frames. Video processing is computationally intensive. We leverage hardware implementations of the H.264 video encoder and decoder available in modern GPUs for decoding and transcoding
    
    
    - transformer-based diffusion models and transformer-based autoregressive models. A diffusion model generates videos by gradually removing noise from a Gaussian noise video. An autoregressive model generates videos piece by piece, conditioned on the past generations following a preset order. Both approaches decompose a difficult video generation problem into easier sub-problems, making it more tractable.
    

- **Ingredients: Blending Custom Photos with Video Diffusion Transformers**
    - (i) a facial extractor that extracts versatile editable facial features for each ID from global and local perspectives;
    - (ii) a multi-scale projector that projects embeddings to the context space of image query;
    - (iii) an ID router that dynamically allocates and integrates the ID embeddings across their respective regions. a routing mechanism that combines and distributes multiple ID embeddings in the context space of image query dynamically. Supervised with classification loss, it avoids blending of various identities and preserving individuality
    - The multi-stage training process sequentially optimizes the ID embedding and ID router components. As a result, our method enables the customization of multiple IDs without prompt constraints, offering great adaptability and precision in video synthesis. The training procedure is systematically divided into two distinct stages: f**acial embedding alignment phase and routing fine-tuning phase.**
    
    
- **TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training**
    - Recent works have addressed this by reducing the number of tokens processed in the model, often through masking. In contrast, this work aims to improve the training efficiency of the diffusion backbone by **using predefined routes that store this information until it is reintroduced to deeper layers of the model, rather than discarding these tokens entirely**.
    - Further, we combine multiple routes and introduce an adapted auxiliary loss that accounts for all applied routes.
    
    
- **Decentralized Diffusion Models**
    - Our method trains a set of expert diffusion models over partitions of the dataset, each in full isolation from one another. At inference time, the experts ensemble through a lightweight router. We show that the ensemble collectively optimizes the sam


- **3DIS-FLUX: simple and efficient multi-instance generation with DiT rendering**
    - decouples MIG (Multi-instance generation) into two distinct phases: 1) depth-based scene construction and 2) detail rendering with widely pre-trained depth control models.


- **An Empirical Study of Autoregressive Pre-training from Videos**
    - We explore different architectural, training, and inference design choices. We evaluate the learned visual representations on a range of downstream tasks including image recognition, video classification, object tracking, and robotics.
    - We treat **videos as sequences of visual tokens and train a causal transformer models on next-token prediction task**. We use causal transformer model with **LLaMa** (Touvron et al., 2023) architecture. We use **dVAE** (Ramesh et al., 2021) to tokenize frames into discrete tokens. Treating videos as sequences of tokens enables us to jointly train on videos and images using a unified format. We construct a diverse dataset of videos and images comprising over 1 trillion visual tokens. Our models are first pre-trained on this data and then evaluated on downstream tasks.
    
    
- **Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Control**
    - Our key insight is that achieving versatile video control necessitates leveraging 3D control signals, as videos are fundamentally 2D renderings of dynamic 3D content. Unlike prior methods limited to 2D control signals, DaS leverages 3D tracking videos as control inputs, making the video diffusion process inherently 3D-aware.
    
    
- **LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token**
    - To achieve a high compression ratio of vision tokens while preserving visual information, we first analyze how LMMs understand vision tokens and **find that most vision tokens only play a crucial role in the early layers** of LLM backbone, where they mainly fuse visual information into text tokens. Building on this finding, LLaVA-Mini introduces modality **pre-fusion to fuse visual information into text tokens in advance**, thereby facilitating the extreme compression of vision tokens fed to LLM backbone into one token.

- **Magic Mirror: ID-Preserved Video Generation in Video Diffusion Transformers**
    - (1) a dual-branch facial feature extractor that captures both identity and structural features
    - (2) a lightweight cross-modal adapter with Conditioned Adaptive Normalization for efficient identity integration
    - (3) a two-stage training strategy combining synthetic identity pairs with video data
    
- **Motion-Aware Generative Frame Interpolation**

- **Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation**
    - (i) An explicit intermediate representation generation stage, followed by (ii) A video generation stage that is conditioned on this representation.
    - Our key innovation is the **introduction** of a mask-based motion trajectory as an intermediate representation, that captures both semantic object information and motion, enabling an expressive but compact representation of motion and semantics.
    - **Image-to-Motion Generation:** In the first stage, outlined in Sec. 3.2, we generate motion trajectory conditioned on the reference image and motion prompt. This motion trajectory encapsulates the dynamic behavior of individual objects.
    - **Motion-to-Video Generation:** In the second stage, outlined in Sec. 3.3, we use the generated motion trajectory, along with the object-specific prompts and the reference image, to produce a photorealistic video.
    
- **EditAR: Unified Conditional Generation with Autoregressive Models**
    - The model takes both images and instructions as inputs, and predicts the edited images tokens in a vanilla next-token paradigm. jointly trained on various image manipulation and image translation tasks, and demonstrates promising potential towards building a unified conditional image generation model
    - EditAR builds primarily on Llamagen [61], a text-toimage autoregressive model based on the Llama2 [64, 65] architecture that has demonstrated impressive image generation capabilities. However, due to the lack of a conditional image input, Llamagen does not support tasks like image manipulation or translation. To allow this, we adapt the architecture by prefilling the model with image tokens from a conditioning input image, along with additional positional embeddings

- **MEMO: Memory-Guided Diffusion for Expressive Talking Video Generation**
    - (1) a memory-guided temporal module, which enhances long-term identity consistency and motion smoothness by developing memory states to store information from a longer past context to guide temporal modeling via linear attention;
    - and (2) an emotion-aware audio module, which replaces traditional cross attention with multimodal attention to enhance audio-video interaction, while detecting emotions from audio to refine facial expressions via emotion adaptive layer norm.
    
    
- **DiCoDe: Diffusion-Compressed Deep Tokens for Autoregressive Video Generation with Language Models**
    - DiCoDe is composed a video diffusion model as the tokenizer to extract deep tokens and a autoregressive language model to predict the sequences of deep tokens.
    - 1) Temporally causal: By encoding video clips in a way that preserves temporal order, DiCoDe aligns with the sequential nature of AR models and video data; 2) Highly compressed: By leveraging the prior knowledge of video diffusion model, videos can be represented with a manageable number of tokens for efficient AR modeling; 3) Compatible with image data: Our frame-level tokenizer allows images to be effectively represented, alleviating the scarcity of high-quality video-text data.
    

- **GENMAC: Compositional Text-to-Video Generation with Multi-Agent Collaboration**
    - an iterative, multi-agent framework that enables compositional text-to-video generation. The collaborative workflow includes three stages: DESIGN, GENERATION, and REDESIGN, with an iterative loop between GENERATION and REDESIGN stages to progressively verify and refine the generated videos. The REDESIGN stage is the most challenging stage that aims to **verify the generated videos, suggest corrections, and redesign the text prompts, frame-wise layouts, and guidance scales for the next iteration of generation**. To avoid hallucination of a single MLLM agent, we decompose this stage to four sequentiallyexecuted MLLM-based agents: verification agent, suggestion agent, correction agen
    
    
- **Enhancing Compositional Text-to-Image Generation with Reliable Random Seeds**
    - Our analyses reveal that **different initial random seeds** tend to guide the model to place objects in distinct image areas, potentially adhering to specific patterns of camera angles and image composition associated with the seed.
    
    
- **Fleximo: Towards Flexible Text-to-Human Motion Video Generation**


- **SPAgent: Adaptive Task Decomposition and Model Selection for General Video Generation and Editing**
    - SPAgent assembles a tool library integrating state-of-the-art open-source image and video generation and editing models as tools. After fine-tuning on our manually annotated dataset, SPAgent can automatically coordinate the tools for video generation and editing, through our novelly designed three-step framework:
    - (1) decoupled intent recognition, (2) principle-guided route planning, and (3) capability-based execution model selection.
        
        
- **Imagine360: Immersive 360 Video Generation from Perspective Anchor**
    - 1) Firstly we adopt the dual-branch design, including a perspective and a panorama video denoising branch to provide local and global constraints for 360◦ video generation, with motion module and spatial LoRA layers fine-tuned on extended web 360◦ videos
    - 2) Additionally, an antipodal mask is devised to capture long range motion dependencies, enhancing the reversed camera motion between antipodal pixels across hemispheres.
    - 3) To handle diverse perspective video inputs, we propose elevation-aware designs that adapt to varying video masking due to changing elevations across frames.
    
- **Diffusion Model with Perceptual Loss**
    - Our analysis suggests that the loss objective has an important role in shaping the learned distribution, and we hypothesize that the common squared distance loss objective is not optimal
    - we propose a novel self-perceptual objective that uses the diffusion model itself as the perceptual loss.
- **Timestep Embedding Tells: It’s Time to Cache for Video Diffusion Model**
    - Timestep Embedding Aware Cache (TeaCache), a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps
    - TeaCache first modulates the noisy inputs using the timestep embeddings to ensure their differences better approximating those of model outputs. TeaCache then introduces a rescaling strategy to refine the estimated differences and utilizes them to indicate output caching.
    
- **AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers**
    - First, we determine that motion induced by camera movements in videos is **low-frequency in nature**. This motivates us to adjust train and test pose conditioning schedules, accelerating training convergence while improving visual and motion quality.
    - Then, by probing the representations of an unconditional video diffusion transformer, we observe that they implicitly perform camera pose estimation under the hood, and **only a sub-portion of their layers contain the camera information**. This suggested us to limit the injection of camera conditioning to a subset of the architecture to prevent interference with other video features, leading to 4× reduction of training parameters, improved training speed and 10% higher visual quality.
    
    
- **From Slow Bidirectional to Fast Autoregressive Video Diffusion Models**
    - We address this limitation by adapting a pretrained bidirectional diffusion transformer to an autoregressive transformer that generates frames on-the-fly. To further reduce latency, we extend distribution matching distillation (DMD) to videos, distilling 50-step diffusion model into a 4-step generator. To enable stable and high-quality distillation, we introduce a student initialization scheme based on teacher's ODE trajectories, as well as an asymmetric distillation strategy that supervises a causal student model with a bidirectional teacher.
    
    
- **Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps**
    - We structure the design space along two axes: the verifiers used to provide feedback, and the algorithms used to find better noise candidates.
    - For verifiers, we consider the three different settings, which are meant to simulate three different use cases: (1) where we have **privileged information** about how the final evaluation is carried out; (2) where we have **conditioning information** for guiding the generation; (3) where we have **no extra information** available. (**Verifiers are used to evaluate the goodness of candidates**)
    - For algorithms, we consider (1) **Random Search**, which simply selects the best from a fixed set of candidates; (2) **Zero-Order Search**, which leverages verifiers feedback to iteratively refine noise candidates; (3) **Search over Paths,** which leverages verifiers feedback to iteratively refine diffusion sampling trajectories.
    
    
    - We examine how different verifier–algorithm combinations perform across various tasks, and our findings indicate that no single configuration is universally optimal; each task instead necessitates a distinct search setup to achieve the best scaling performance.
    - To investigate this, we take two models with good learned representations, CLIP [58] and DINO [53]. Since we only have class labels as the conditioning information on ImageNet, we utilize the classification perspective of the two models. For CLIP, we follow Radford et al. [58] and use the embedding weight generated via prompt engineering3 as a zero-shot classifier. For DINO, we directly take the pre-trained linear classification head. During search, we run samples through the classifiers and select the ones with the highest logits corresponding to the class labels used in generation.
    - Identifying verifiers and algorithms as two crucial design axes in our search framework, we show that optimal configurations vary by task, with no universal solution. Additionally, our investigation into the alignment between different verifiers and generation tasks uncovers their inherent biases, highlighting the need for more carefully designed verifiers to align with specific vision generation tasks.
- **AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark**
    - To be specific, we gradually combine similar tokens in a transformer layer using a bipartite soft matching algorithm to reduce the number of visual tokens.
    - we present VDC, the first benchmark for detailed video captioning, featuring over one thousand videos with significantly longer and more detailed captions.

    
- **VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models**
    - **Motivation**: We show that this limitation arises from the conventional pixel reconstruction objective, which biases models toward appearance fidelity at the expense of motion coherence. **This implies that the model fails to distinguish between a valid video and a temporally incoherent one.**

    
    - VideoJAM is constructed of two units; (a) Training. Given an input video x1 and its motion representation d1, both signals are noised and embedded to a single, joint latent representation using a linear layer, Win+. The diffusion model processes the input, and two linear projection layers predict **both appearance and motion （训练的时候抽取的optical-flow） from the joint representation (Wout+)**. (b) Inference. We propose Inner-Guidance, where the model’s own noisy motion prediction is used to guide the video prediction at each step.
    
    - VideoJAM-4B was fine-tuned using 32 A100 GPUs with a batch size of 32 for 50, 000 iterations on a spatial resolution of 256 × 256. It has a latent dimension of 3072 and 32 attention blocks (same as the base model). VideoJAM-30B was fine-tuned using 256 A100 GPUs with a batch size of 256 for 35, 000 iterations on a spatial resolution of 256 × 256. It has a latent dimension of 6144 and 48 attention blocks (same as the base model). Each attention block is constructed of a self-attention layer that performs spatiotemporal attention between all the video tokens, and a cross-attention layer that integrates the text. Both models were trained with a fixed learning rate of 5e−6, using the Flow Matching paradigm
- **Improved Video VAE for Latent Video Diffusion Model**
    - (1) The initialization from a well-trained image VAE with the same latent dimensions suppresses the improvement of subsequent temporal compression capabilities. (2) The adoption of causal reasoning leads to unequal information interactions and unbalanced performance between frames.
    - To alleviate these problems, we propose a keyframe-based temporal compression (KTC) architecture and a group causal convolution (GCConv) module to further improve video VAE (IV-VAE).
    - the KTC architecture divides the latent space into two branches, in which one half completely inherits the compression prior of keyframes from lower-dimension image VAEs while the other half involves temporal compression into the 3D group causal convolution, reducing temporal-spatial conflicts and accelerating the convergence speed of video VAE
    - The GCConv in above 3D half uses standard convolution within each frame group to ensure inter-frame equivalence, and employs causal logical padding between groups to maintain flexibility in processing variable frame video