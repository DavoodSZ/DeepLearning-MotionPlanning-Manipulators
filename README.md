# Towards Generalist Neural Motion Planners for Robotic Manipulators: Challenges and Opportunities
<a href="https://davoodsz.github.io/planning-manip-survey.github.io/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2408.12831"><strong>Paper</strong></a>
  |
  
<a href="https://zh.engr.tamu.edu/people-2/">Davood Soleymanzadeh</a>,
<a href="https://haosu-robotics.github.io/people">Ivan Lopez-Sanchez</a>,
<a href="https://mae.ncsu.edu/people/hao-su/">Hao Su</a>,
<a href="https://yunzhuli.github.io/">Yunzhu Li</a>,
<a href="https://engineering.tamu.edu/civil/profiles/liang-xiao">Xiao Liang</a>,
<a href="https://engineering.tamu.edu/mechanical/profiles/zheng-minghui">Minghui Zheng</a>

IEEE Transactions on Automation Science and Engineering (T-ASE) (2026)

<p align="center">
<img width="1000" src="./assets/SurveyStructure.svg">
<br>
<em>Fig 1. Review structure with reference to sections of the original paper. Figures are adopted from Carvalho et al. [1], Qureshi et al. [2], Bency et al. [3], and Song et al. [4].</em>
</p>

# Paper List in the Survey Paper
Here is the list of papers we reviewed in our survey paper. We only list the papers presented in "Section V. Deep Learning in Planning for Robotic Manipulators" of the paper. The papers are listed based on the taxonomy in section V of the original paper.

- [End-to-end Planning](#end-to-end-planning): These papers utilize various deep learning frameworks for end-to-end Motion planning for robotic manipulators.
- [Sampling-based Motion Planning](#sampling-based-motion-planning): This group of papers utilizes deep learning to improve the informed sampling primitive and steering primitive of classical sampling-based motion planning algorithms.
- [Constrained Sampling-based Motion Planning](#constrained-sampling-based-motion-planning): This group of papers utilizes deep learning for learning the geometric constraint manifolds for effective on-manifold sample generation.
- [Trajectory Optimization](#trajectory-optimization): This group of papers utilizes deep learning to warm-start the global trajectory optimization problem.
- [Collision Checking](#collision-checking): This group of papers utilizes deep learning for collision querying and collision checking within classical motion planning algorithms.

This list will be continuously updated.

## End-to-end Planning 
- MLPs: **Learning To Find Shortest Collision-Free Paths From Images**, 2020, [Paper Link](https://d1wqtxts1xzle7.cloudfront.net/83099921/2011.14787v1-libre.pdf?1648935949=&response-content-disposition=inline%3B+filename%3DLearning_To_Find_Shortest_Collision_Free.pdf&Expires=1734048774&Signature=TF0BRAHvqZi7oEIlGux7uuxjRteaiaelbXVdGbZEPbJZScRDidoPj6yr8jNd6tzI97hgz7mQ9yfVWhxQIby4lMl16z8HjhEy45ldLm6IrRkOfcvi9-HTUo1eYWpVIOvRWL0EWZKoWk6FGv6HaN07fjGyMXX0Y1eh~4OlDko8DXG-fCoJAxQ-DNMO6NyD6S1AqQzEMimfwkdqGI4AMrzQhAWu5sR13ENta5t76~2fNNqYACs8oGpQGJ1I4peCx58~eAgdTtfoWJJ8-4Sk2BoHTWqZYq5ApSURudZf9VvlZS~fOcjDk~aDyOjt17bktI12Ua~CU0DtiNdNaUTw385GyA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA).
- CNNs: **Physics-informed neural motion planning on constraint manifolds**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10610883).
- CNNs: **Deep reactive planning in dynamic environments**, 2021, [Paper Link](https://proceedings.mlr.press/v155/ota21a/ota21a.pdf)
- CNNs: **Ntfields: Neural time fields for physics-informed robot motion planning**, 2022, [Paper Link](https://arxiv.org/pdf/2210.00120).
- CNNs: **Progressive learning for physics-informed neural motion planning**,2023,[Paper Link](https://arxiv.org/pdf/2306.00616).
- CNNs: **Physics-informed Neural Networks for Robot Motion under Constraints**, 2024,[Paper Link](https://openreview.net/pdf?id=gLf0PnhEO2).
- CNNs: **Physics-informed neural mapping and motion planning in unknown environments**, 2025,[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10916504).
- CNNs: **Physics-informed Temporal Difference Metric Learning for Robot Motion Planning**,2025,[Paper Link](https://arxiv.org/pdf/2505.05691).
- PCNets: **Motion policy networks**, 2023, [Paper Link](https://proceedings.mlr.press/v205/fishman23a/fishman23a.pdf).
- PCNets: **Neural mp: A generalist neural motion planner**, 2024,[Paper Link](https://arxiv.org/abs/2409.05864).
- PCNets: **Deep reactive policy: Learning reactive manipulator motion planning for dynamic environments**, 2025, [Paper Link](https://arxiv.org/pdf/2509.06953?).
- PCNets: **PerFACT: Motion Policy with LLM-Powered Dataset Synthesis and Fusion Action-Chunking Transformers**, 2025, [Paper Link](https://arxiv.org/pdf/2512.03444).
- PCNets: **Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning**, 2024,[Paper Link](https://openreview.net/pdf?id=ccxDydaAs1).
- RNNs: **Neural path planning: Fixed time, near-optimal path generation via oracle imitation**, 2019,[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8968089).
- DGMs - VAEs: **Reaching through latent space: From joint statistics to path planning in manipulation**, 2022,[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9718343).
- DGMs - VAEs: **Leveraging scene embeddings for gradient-based motion planning in latent space**, 2023,[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10161427).
- Foundation Models: **Roco: Dialectic multi-robot collaboration with large language models**, 2024,[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10610855).
- Foundation Models: **Language models as zero-shot trajectory generators**, 2024,[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10549793).
- Foundation Models: **Reshaping robot trajectories using natural language commands: A study of multi-modal data alignment using transformers**, 2022,[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9981810).
- Foundation Models: **Latte: Language trajectory transformer**, 2023,[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10161068).


## Sampling-based Motion Planning 
### Sampling Primitive
- MLPs: **Motion planning networks**, 2019, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8793889).
- MLPs: **Motion planning networks: Bridging the gap between learning-based and classical motion planners**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9154607).
- MLPs: **Neural manipulation planning on constraint manifolds**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9143433).
- MLPs: **Constrained motion planning networks x**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9501956).
- MLPs: **Deeply informed neural sampling for robot motion planning**, 2018, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8593772).
- MLPs: **End-to-end deep learning-based framework for path planning and collision checking: bin-picking application**, 2024, [Paper Link](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D68F8B21FDA8DC6DC3E1D5D312C3178B/S0263574724000109a.pdf/endtoend_deep_learningbased_framework_for_path_planning_and_collision_checking_binpicking_application.pdf).
- MLPs: **A data-driven approach for motion planning of industrial robots controlled by high-level motion commands**, 2023, [Paper Link](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.1030668/full).
- MLPs: **Learning motion planning functions using a linear transition in the c-space: Networks and kernels**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9529848).
- MLPs: **Parallelised diffeomorphic sampling-based motion planning**, 2022, [Paper Link](https://proceedings.mlr.press/v164/lai22a/lai22a.pdf)
- MLPs: **Leveraging experience in lazy search**, 2021, [Paper Link](https://link.springer.com/content/pdf/10.1007/s10514-021-10018-5.pdf).
- MLPs: **Motion planning of manipulator by points-guided sampling network**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9765528).
- CNNs: **Using deep learning to bootstrap abstractions for hierarchical robot planning**, 2022, [Paper Link](https://arxiv.org/pdf/2202.00907).
- CNNs: **Prediction of bottleneck points for manipulation planning in cluttered environment using a 3d convolutional neural network**, 2019, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8988766).
- CNNs: **A Hybrid AI-based Adaptive Path Planning for Intelligent Robot Arms**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10339309).
- CNNs: **Learning to retrieve relevant experiences for motion planning**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9812076).
- CNNs: **3d-cnn based heuristic guided task-space planner for faster motion planning**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9196883).
- RNNs: **A data-driven approach for motion planning of industrial robots controlled by high-level motion commands**, 2023, [Paper Link](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.1030668/full).
- RNNs: **Deep learning-based optimization for motion planning of dual-arm assembly robots**, 2021, [Paper Link](https://www.sciencedirect.com/science/article/pii/S0360835221005076).
- GNNs: **KG-Planner: Knowledge-Informed Graph Neural Planning for Collaborative Manipulators**, 2024, [Paper Link](https://arxiv.org/pdf/2405.07962).
- GNNs: **Integrating Uncertainty-Aware Human Motion Prediction Into Graph-Based Manipulator Motion Planning**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10550073).
- GNNs: **SIMPNet: Spatial-Informed Motion Planning Network**, 2024, [Paper Link](https://arxiv.org/pdf/2408.12831).
- GNNs: **Reducing collision checking for sampling-based motion planning using graph neural networks**, 2021, [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2021/file/224e5e49814ca908e58c02e28a0462c1-Paper.pdf).
- GNNs: **Hardware architecture of graph neural network-enabled motion planner**, 2022, [Paper Link](https://dl.acm.org/doi/pdf/10.1145/3508352.3561113).
- GNNs: **Learning-based motion planning in dynamic environments using gnns and temporal encoding**, 2022, [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1d4798259250f2b4fe38614b48f8996-Paper-Conference.pdf).
- GNNs: **Dyngmp: Graph neural network-based motion planning in unpredictable dynamic environments**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10342326).
- DGMs - VAEs: **Motion planning networks**, 2019, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8793889).
- DGMs - VAEs: **Motion planning networks: Bridging the gap between learning-based and classical motion planners**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9154607).
- DGMs - VAEs: **Deeply informed neural sampling for robot motion planning**, 2018, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8593772).
- DGMs - VAEs: **Robot motion planning in learned latent spaces**, 2019, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8653875).
- DGMs - VAEs: **Motion planning of manipulator by points-guided sampling network**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9765528).
- DGMs - VAEs: **Learning sampling distributions for robot motion planning**, 2018, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8460730).
- DGMs - VAEs: **SERA: Safe and Efficient Reactive Obstacle Avoidance for Collaborative Robotic Planning in Unstructured Environments**, 2022, [Paper Link](https://arxiv.org/pdf/2203.13821).
- DGMs - VAEs: **Reactive Whole-Body Obstacle Avoidance for Collision-Free Human-Robot Interaction with Topological Manifold Learning.**, 2022, [Paper Link](https://d1wqtxts1xzle7.cloudfront.net/94127094/2203.13821-libre.pdf?1668284615=&response-content-disposition=inline%3B+filename%3DReactive_Whole_Body_Obstacle_Avoidance_f.pdf&Expires=1734110864&Signature=HVICuj8KGJMuHGaVQdznVrkHwecszB9au2YOg-8WzzNNMirmbqSiDxlaJtDsx08TlNoNm~Ir90xsRXkmf0m5GA4LfM~fYmyf7erzejvR-JvqFSRoALmwlQKdj8dPy4BS2WKqUUHX9dyxwutYWZOorAEBXLw-HRh9HoXc6rwFHvZ9EB2bLDW03nky43yQQfNyF9e2WvwUcDpV2NtHHb-uP65G5nwgIvq3qvPwIBxIakZNTQb-eIGXKN1NdqFEG0nE2B5By6K73OyZzZ4Iz73aSacx061D4WRHzqvsITyOnhhAX5zTB9rE2enHZukzhdA13WjQvU-jeC84AUSRQZbLxQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA).
- DGMs - VAEs: **DAMON: Dynamic Amorphous Obstacle Navigation using Topological Manifold Learning and Variational Autoencoding**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10342035).
- DGMs - VAEs: **Intercepting A Flying Target While Avoiding Moving Obstacles: A Unified Control Framework With Deep Manifold Learning**, 2022, [Paper Link](https://arxiv.org/pdf/2209.13628).
- DGMs - VAEs: **LEGO: Leveraging experience in roadmap generation for sampling-based planning**, 2019, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8968503).
- DGMs - VAEs: **Robotic motion planning using learned critical sources and local sampling**, 2020, [Paper Link](https://arxiv.org/pdf/2006.04194).
- DGMs - VAEs: **Learning from Local Experience: Informed Sampling Distributions for High Dimensional Motion Planning**, 2023, [Paper Link](https://arxiv.org/pdf/2306.09446).
- DGMs - VAEs: **Neural randomized planning for whole body robot motion**, 2024, [Paper Link](https://arxiv.org/pdf/2405.11317).
- DGMs - VAEs: **Planning with Learned Subgoals Selected by Temporal Information**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10610538).
- DGMs - VAEs: **Reaching through latent space: From joint statistics to path planning in manipulation**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9718343).
- DGMs - VAEs: **Leveraging scene embeddings for gradient-based motion planning in latent space**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10161427).
- DGMs - VAEs: **Predicting sample collision with neural networks**, 2020, [Paper Link](https://arxiv.org/pdf/2006.16868).
- DGMs - VAEs: **Learning sampling dictionaries for efficient and generalizable robot motion planning with transformers**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10271531).
- DGMs - VAEs: **Zero-Shot Constrained Motion Planning Transformers Using Learned Sampling Dictionaries**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10611398).
- DGMs - VAEs: **Graph Wasserstein autoencoder-based asymptotically optimal motion planning with kinematic constraints for robotic manipulation**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9708428).
- DGMs - GANs: **Guiding search in continuous state-action spaces by learning an action sampler from off-target search experience**, 2018, [Paper Link](https://ojs.aaai.org/index.php/AAAI/article/view/12106/11965).
- DGMs - GANs: **Learning-based collision-free planning on arbitrary optimization criteria in the latent space through cGANs**, 2023, [Paper Link](https://www.tandfonline.com/doi/pdf/10.1080/01691864.2023.2180327).
- DGMs - NFs: **Plannerflows: Learning motion samplers with normalising flows**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9636190).
- Transformers: **Learning to plan in high dimensions via neural exploration-exploitation trees**, 2019. [Paper Link](https://arxiv.org/pdf/1903.00070).

### Steering Primitive
- MLPs: **Fast swept volume estimation with deep learning**, 2020, [Paper Link](https://www.cs.unm.edu/tapialab/Publications/60.pdf).
- MLPs: **Fast deep swept volume estimator**, 2021, [Paper Link](https://journals.sagepub.com/doi/pdf/10.1177/0278364920940781).
- MLPs: **Multitask and Transfer Learning of Geometric Robot Motion**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9636052).

## Constrained Sampling-based Motion Planning
- MLPs: **Learning equality constraints for motion planning on manifolds**, 2021, [Paper Link](https://proceedings.mlr.press/v155/sutanto21a/sutanto21a.pdf).
- MLPs: **Fast kinodynamic planning on the constraint manifold with deep neural networks**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10292912).
- MLPs: **Ntfields: Neural time fields for physics-informed robot motion planning**, 2022, [Paper Link](https://arxiv.org/pdf/2210.00120).
- MLPs: **Progressive Learning for Physics-informed Neural Motion Planning**, 2023, [Paper Link](https://arxiv.org/pdf/2306.00616).
- MLPs: **Physics-informed Neural Networks for Robot Motion under Constraints**, 2024, [Paper Link](https://openreview.net/pdf?id=gLf0PnhEO2).
- MLPs: **Physics-informed Neural Mapping and Motion Planning in Unknown Environments**, 2024, [PaperLink](https://arxiv.org/pdf/2410.09883).
- DGMs - VAEs: **LAC-RRT: Constrained Rapidly-Exploring Random Tree with Configuration Transfer Models for Motion Planning**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10244038).
- DGMs - VAEs: **A Constrained Motion Planning Method Exploiting Learned Latent Space for High-Dimensional State and Constraint Spaces**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10538476).
- DGMs - GANs: **Generative adversarial network to learn valid distributions of robot configurations for inverse kinematics and constrained motion planning**, 2020, [Paper Link](https://www.researchgate.net/profile/Teguh-Lembono/publication/345756820_Generative_Adversarial_Network_to_Learn_Valid_Distributions_of_Robot_Configurations_for_Inverse_Kinematics_and_Constrained_Motion_Planning/links/5fc0ca3992851c933f6606d6/Generative-Adversarial-Network-to-Learn-Valid-Distributions-of-Robot-Configurations-for-Inverse-Kinematics-and-Constrained-Motion-Planning.pdf).
- DGMs - GANs: **Learning constrained distributions of robot configurations with generative adversarial network**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9385935).
- DGMs - GANs: **Approximating constraint manifolds using generative models for sampling-based constrained motion planning**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9561456).

## Trajectory Optimization
- MLPs: **Deep learning can accelerate grasp-optimized motion planning**, 2020, [Paper Link](https://www.science.org/doi/pdf/10.1126/scirobotics.abd7710).
- MLPs: **Learning-based warm-starting for fast sequential convex programming and trajectory optimization**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9172293).
- DGMs - VAEs: **Motion planning by learning the solution manifold in trajectory optimization**, 2022, [Paper Link](https://journals.sagepub.com/doi/pdf/10.1177/02783649211044405).
- DGMs - DMs: **Motion planning diffusion: Learning and planning of robot motions with diffusion models**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10342382).
- DGMs - DMs: **Motion Planning Diffusion: Learning and Adapting Robot Motion Planning with Diffusion Models**, 2024, [Paper Link](https://arxiv.org/pdf/2412.19948).
- DGMs - DMs: **Language-guided object-centric diffusion policy for collision-aware robotic manipulation**, 2024, [Paper Link](https://arxiv.org/pdf/2407.00451).
- DGMs - DMs: **M2diffuser: Diffusion-based trajectory optimization for mobile manipulation in 3d scenes**, 2024, [Paper Link](https://arxiv.org/pdf/2410.11402).
- DGMs - DMs: **Diffusion-based generation, optimization, and planning in 3d scenes**, 2023, [Paper Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Diffusion-Based_Generation_Optimization_and_Planning_in_3D_Scenes_CVPR_2023_paper.pdf).
- DGMs - DMs: **Sampling constrained trajectories using composable diffusion models**, 2023, [Paper Link](https://openreview.net/pdf?id=UAylEpIMNE).
- DGMs - DMs: **Efficient and Guaranteed-Safe Non-Convex Trajectory Optimization with Constrained Diffusion Model**, 2024, [Paper Link](https://arxiv.org/pdf/2403.05571).
- DGMs - DMs: **Constraint-Aware Diffusion Models for Trajectory Optimization**, 2024, [Paper Link](https://arxiv.org/pdf/2406.00990).
- DGMs - DMs: **APEX: Ambidextrous Dual-Arm Robotic Manipulation Using Collision-Free Generative Diffusion Models**, 2024, [Paper Link](https://arxiv.org/pdf/2404.02284).
- DGMs - DMs: **Edmp: Ensemble-of-costs-guided diffusion for motion planning**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10610519).
- DGMs - DMs: **Potential based diffusion motion planning**, 2024, [Paper Link](https://arxiv.org/pdf/2407.06169).
- DGMs - EBMs: **Learning implicit priors for motion optimization**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9981264).
- DGMs - EBMs: **Global and reactive motion generation with geometric fabric command sequences**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10160965).

## Collision Checking
- MLPs: **Robot motion planning in learned latent spaces**, 2019, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8653875).
- MLPs: **Neural collision clearance estimator for batched motion planning**, 2020, [Paper Link](https://arxiv.org/pdf/1910.05917).
- MLPs: **Whole-body Self-collision Distance Detection for A Heavy-duty Manipulator Using Neural Networks**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10356839).
- MLPs: **Reinforcement learning in robotic motion planning by combined experience-based planning and self-imitation learning**, 2023, [Paper Link](https://www.sciencedirect.com/science/article/pii/S0921889023001847).
- MLPs: **Comparison of machine learning techniques for self-collisions checking of manipulating robots**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10242571).
- MLPs: **Active learning of the collision distance function for high-DOF multi-arm robot systems**, 2024, [Paper Link](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/A89DA2FCB73EBCC6FA1EC00E3DC53AF4/S0263574723001790a.pdf/div-class-title-active-learning-of-the-collision-distance-function-for-high-dof-multi-arm-robot-systems-div.pdf).
- MLPs: **Implicit Distance Functions: Learning and Applications in Control**, 2020, [Paper Link](https://infoscience.epfl.ch/server/api/core/bitstreams/80520f75-bd5d-4617-90d7-325411219398/content).
- MLPs: **Neural joint space implicit signed distance functions for reactive robot manipulator control**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9976191).
- MLPs: **Reactive collision-free motion generation in joint space via dynamical systems and sampling-based MPC**, 2024, [Paper Link](https://journals.sagepub.com/doi/pdf/10.1177/02783649241246557).
- MLPs: **Regularized deep signed distance fields for reactive motion generation**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9981456).
- MLPs: **Collision-free motion generation based on stochastic optimization and composite signed distance field networks of articulated robot**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10238810).
- MLPs: **Stochastic implicit neural signed distance functions for safe motion planning under sensing uncertainty**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10610773).
- MLPs: **Configuration Space Distance Fields for Manipulation Planning**, 2024, [Paper Link](https://arxiv.org/pdf/2406.01137).
- MLPs: **Deep prediction of swept volume geometries: Robots and resolutions**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9341396).
- MLPs: **Single swept volume reconstruction by signed distance function learning: A feasibility study based on implicit geometric regularization**, 2022, [Paper Link](https://homepage.iis.sinica.edu.tw/papers/liu/25695-F.pdf).
- MLPs: **Reachability-based trajectory design with neural implicit safety constraints**, 2023, [Paper Link](https://arxiv.org/pdf/2302.07352).
- MLPs: **Neural Implicit Swept Volume Models for Fast Collision Detection**, 2024, [Paper Link](https://arxiv.org/pdf/2402.15281).
- MLPs: **Reliable and Accurate Implicit Neural Representation of Multiple Swept Volumes with Application to Safe Human–Robot Interaction**, 2024, [Paper Link](https://link.springer.com/content/pdf/10.1007/s42979-024-02640-8.pdf).
- MLPs: **Fast Collision Detection for Robot Manipulator Path: an Approach Based on Implicit Neural Representation of Multiple Swept Volumes**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10268533).
- CNNs: **Object rearrangement using learned implicit collision functions**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9561516).
- CNNs: **Cabinet: Scaling neural collision detection for object rearrangement with procedural scene generation**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10161528).
- GNNs: **Graph-based 3D Collision-distance Estimation Network with Probabilistic Graph Rewiring**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10611465).
- GNNs: **GraphDistNet: A graph-based collision-distance estimator for gradient-based trajectory optimization**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9851942).
- GNNs: **PairwiseNet: Pairwise Collision Distance Learning for High-dof Robot Systems**, 2023, [Paper Link](https://proceedings.mlr.press/v229/kim23d/kim23d.pdf).
- Transformers: **GraphDistNet: A graph-based collision-distance estimator for gradient-based trajectory optimization**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9851942).
- Transformers: **DistFormer: A High-Accuracy Transformer-Based Collision Distance Estimator for Robotic Arms**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10245243).

# Acknowledgement
The structure of this repository is based on the repository of the survey paper titled: [Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis](https://robotics-fm-survey.github.io/).

Please contact [Davood Soleymanzadeh](https://zh.engr.tamu.edu/people-2/) with any questions or suggestions.

# References
1. [Motion planning diffusion: Learning and planning of robot motions with diffusion models](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10342382&casa_token=v5emJzVw7rsAAAAA:AkG7symAkhkSMl0bL3Y9TUZCRfb4tZXaqCGDzzc9L5AeuudFwwD3JFe9v9nhcTcfDw-cuxSG)
2. [Motion planning networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8793889&casa_token=KEQqA8lfWacAAAAA:mHBn2XI7S6gpigWvWmhayI-280Rz4TAuPkGua4y2f0gmOYowHyK-EE4ZW_x1QPcTo1wdvBa-)
3. [Neural path planning: Fixed time, near-optimal path generation via oracle imitation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8968089&casa_token=R0o0ErTXOcsAAAAA:x1veVh7vOQz0KyAr80Le19jVSIakx-RXSugnj9ktEzRjiHCZ_B_uYD1cwpWqRLwXx4V6McXY)
4. [Graph-based 3D Collision-distance Estimation Network with Probabilistic Graph Rewiring](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10611465&casa_token=69VgilAGhGgAAAAA:YXf9XFNtHgLOLHGzPKCHW5yG07s7hOSN0ShxKfwsfxAE1jXG05wl1mRZw6AOSnv5KBpaPTVT)

# Citation
If you find our work useful, please consider citing:

```
```
