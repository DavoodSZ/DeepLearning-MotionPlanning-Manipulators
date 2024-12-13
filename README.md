# Paper List in the Survey Paper
Here is the list of papers we reviewed in our survey paper. We only list the papers presented in "Section V. Deep Learning in Planning for Robotic Manipulators" of the paper. The papers are listed based on the taxonomy in section V of the original paper.

- [End-to-end Planning](#end-to-end-planning): These papers utilize various deep learning frameworks for end-to-end Motion planning for robotic manipulators.
- [Sampling-based Motion Planning](#sampling-based-motion-planning): This group of papers utilizes deep learning to improve the informed sampling primitive and steering primitive of classical sampling-based motion planning algorithms.
- [Constrained Sampling-based Motion Planning](#constrained-sampling-based-motion-planning): This group of papers utilizes deep learning for learning the geometric constraint manifolds for effective on-manifold sample generation.
- [Trajectory Optimization](#trajectory-optimization): This group of papers utilizes deep learning to warm-start the global trajectory optimization problem.
- [Collision Checking](#collision-checking): This group of papers utilizes deep learning for collision querying and collision checking within classical motion planning algorithms.

This list will be continuously updated.

## End-to-end Planning 
- **Learning To Find Shortest Collision-Free Paths From Images**, 2020, [Paper Link](https://d1wqtxts1xzle7.cloudfront.net/83099921/2011.14787v1-libre.pdf?1648935949=&response-content-disposition=inline%3B+filename%3DLearning_To_Find_Shortest_Collision_Free.pdf&Expires=1734048774&Signature=TF0BRAHvqZi7oEIlGux7uuxjRteaiaelbXVdGbZEPbJZScRDidoPj6yr8jNd6tzI97hgz7mQ9yfVWhxQIby4lMl16z8HjhEy45ldLm6IrRkOfcvi9-HTUo1eYWpVIOvRWL0EWZKoWk6FGv6HaN07fjGyMXX0Y1eh~4OlDko8DXG-fCoJAxQ-DNMO6NyD6S1AqQzEMimfwkdqGI4AMrzQhAWu5sR13ENta5t76~2fNNqYACs8oGpQGJ1I4peCx58~eAgdTtfoWJJ8-4Sk2BoHTWqZYq5ApSURudZf9VvlZS~fOcjDk~aDyOjt17bktI12Ua~CU0DtiNdNaUTw385GyA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA).
- **Deep reactive planning in dynamic environments**, 2021. [Paper Link](https://proceedings.mlr.press/v155/ota21a/ota21a.pdf).
- **Motion policy networks**, 2023, [Paper Link](https://proceedings.mlr.press/v205/fishman23a/fishman23a.pdf).
- **Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning**, 2024, [Paper Link](https://openreview.net/pdf?id=ccxDydaAs1).
- **Neural mp: A generalist neural motion planner**, 2024, [Paper Link](https://arxiv.org/abs/2409.05864).
- **Transformer-Enhanced Motion Planner: Attention-Guided Sampling for State-Specific Decision Making**, 2024, [Paper Link](https://arxiv.org/pdf/2404.19403)
- **Roco: Dialectic multi-robot collaboration with large language models**, 2024, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10610855).

## Sampling-based Motion Planning 
### Sampling Primitive
- MLP: **Motion planning networks**, 2019, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8793889).
- MLP: **Motion planning networks: Bridging the gap between learning-based and classical motion planners**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9154607).
- MLP: **Neural manipulation planning on constraint manifolds**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9143433).
- MLP: **Constrained motion planning networks x**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9501956).
- MLP: **Deeply informed neural sampling for robot motion planning**, 2018, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8593772).
- MLP: **End-to-end deep learning-based framework for path planning and collision checking: bin-picking application**, 2024, [Paper Link](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D68F8B21FDA8DC6DC3E1D5D312C3178B/S0263574724000109a.pdf/endtoend_deep_learningbased_framework_for_path_planning_and_collision_checking_binpicking_application.pdf).
- MLP: **A data-driven approach for motion planning of industrial robots controlled by high-level motion commands**, 2023, [Paper Link](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.1030668/full).
- MLP: **Learning motion planning functions using a linear transition in the c-space: Networks and kernels**, 2021, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9529848).
- MLP: **Parallelised diffeomorphic sampling-based motion planning**, 2022, [Paper Link](https://proceedings.mlr.press/v164/lai22a/lai22a.pdf)
- MLP: **Leveraging experience in lazy search**, 2021, [Paper Link](https://link.springer.com/content/pdf/10.1007/s10514-021-10018-5.pdf).
- CNN: **Using deep learning to bootstrap abstractions for hierarchical robot planning**, 2022, [Paper Link](https://arxiv.org/pdf/2202.00907).
- CNN: **Prediction of bottleneck points for manipulation planning in cluttered environment using a 3d convolutional neural network**, 2019, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8988766).
- CNN: **A Hybrid AI-based Adaptive Path Planning for Intelligent Robot Arms**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10339309).
- CNN: **Learning to retrieve relevant experiences for motion planning**, 2022, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9812076).
- CNN: **3d-cnn based heuristic guided task-space planner for faster motion planning**, 2020, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9196883).
- RNN: **A data-driven approach for motion planning of industrial robots controlled by high-level motion commands**, 2023, [Paper Link](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.1030668/full).
- RNN: **Deep learning-based optimization for motion planning of dual-arm assembly robots**, 2021, [Paper Link](https://www.sciencedirect.com/science/article/pii/S0360835221005076).
- GNN: **SIMPNet: Spatial-Informed Motion Planning Network**, 2024, [Paper Link](https://arxiv.org/pdf/2408.12831).
- GNN: **Reducing collision checking for sampling-based motion planning using graph neural networks**, 2021, [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2021/file/224e5e49814ca908e58c02e28a0462c1-Paper.pdf).
- GNN: **Hardware architecture of graph neural network-enabled motion planner**, 2022, [Paper Link](https://dl.acm.org/doi/pdf/10.1145/3508352.3561113).
- GNN: **Learning-based motion planning in dynamic environments using gnns and temporal encoding**, 2022, [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1d4798259250f2b4fe38614b48f8996-Paper-Conference.pdf).
- GNN: **Dyngmp: Graph neural network-based motion planning in unpredictable dynamic environments**, 2023, [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10342326).
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

## Trajectory Optimization

## Collision Checking
