# ChurnSubgraph
Official repo for paper "When Subgraphs Outperform Graphs: A Scalable Training Strategy for Churn Prediction on Large Class-imbalanced Networks".

Abstract: Churn prediction, due to its imbalanced class distribution and high precision requirements on top-ranked instances, is particularly challenging among prediction tasks. Although in recent years, GNNs (Graph Neural Networks) have shown promise for churn prediction by leveraging user interactions, existing graph-based methodologies for handling class imbalance still encounter significant limitations regarding computational costs, structural distortion caused by synthetic or rewired edges, and incompatibility with inductive (near) real-time inference settings. To address these limitations, we propose a scalable subgraph training strategy that treats each node-centric subgraph as an independent training instance. This framework naturally supports inductive learning, avoids costly full-graph augmentation, and enables seamless integration with a wide range of class imbalance-handling techniques. To validate the efficiency of our approach, we conduct an exhaustive comparison against state-of-the-art full-graph methods across multiple GNN architectures on three large-scale, real-world telecommunications graphs consisting of millions of nodes. The results demonstrate that the subgraph strategy outperforms full-graph alternatives in terms of both AUC-PR and lift whilst offering better scalability and robustness.

The experiments were conducted with reference to the following repository, and all related works have been cited in the references.

[PyTorch Learning to Rank (LTR)](https://github.com/rjagerman/pytorchltr).

[GraphME](https://github.com/12chen20/GraphME).

[BalancedMetaSoftmax - Classification](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/tree/main).

[BAT: BAlanced Topological augmentation](https://github.com/ZhiningLiu1998/BAT?tab=readme-ov-file).

[TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification](https://github.com/Jaeyun-Song/TAM/tree/main).


