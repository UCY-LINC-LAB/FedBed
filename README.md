# FedBed
Benchmarking Federated Learning over Virtualized Edge Testbeds


Federated Learning has become the de facto paradigm for training AI models under a distributed modality where the computational effort is spread across several clients without sharing local data. Despite its distributed nature, enabling FL in an Edge-Cloud continuum is challenging with resource and network heterogeneity, different AI models and libraries, and non-uniform data distributions, all hampering QoS and limiting innovation potential. This work introduces FedBed, a  testing framework that enables the rapid and reproducible benchmarking of FL deployments on virtualized testbeds. FedBed aids users in assessing the numerous trade-offs that result from combining a variety of FL software and infrastructure configurations in Edge-Cloud settings. This reduces the time-consuming process that includes the setup of either a virtual physical or emulation testbed, experiment configurations, and the monitoring of the resulting FL testbed. 

### System Model & Configurations

FedBed considers an FL deployment as a multilayered system, as depicted in the following Figure, with: (i) *FL Training Layer* being deployed on top, responsible for the FL training; (ii) *Data Layer* that is characterized by its dataset distribution; and (iii) *Execution Layer*, which illustrates the underlying compute and network resources. 

![System Model and Configurations](https://github.com/UCY-LINC-LAB/FedBed/blob/main/figures/SystemModel.png?raw=true)

### FedBed Overview

Users face increased challenges mentioned above when executing FL workloads on Edge devices, and seek solutions to automatically evaluate various FL aspects such as ML models, aggregation algorithms, infrastructure properties, and more. 
To address these challenges, the FedBed testing framework allows users to choose their desired combination of built-in ML models, datasets, and aggregation algorithms, making it easy for them to evaluate the FL performance and infrastructure implications with minimal effort.
The only precondition for users is to have an already installed virtual testbed orchestrator compatible with the FedBed framework.

The following figure provides a high-level overview of the framework and its functionalities. Users start the evaluation by designing the FedBed's composable FL model in a YAML file and submitting it to the *FedBed Interface*, which is a Python light-weight library. 
FedBed Interface can be utilized from interactive data analysis tools, e.g., Jupyter Notebooks, and users can re-use the model's YAML file and the notebooks to easily reproduce their experimental analysis.
With FL model submitted, FedBed validates model's parameters and the available resources, and propagates the model to FedBed Controller.

<p align="center">
  <img src="https://github.com/UCY-LINC-LAB/FedBed/blob/main/figures/overview.png?raw=true" alt="System Model and Configurations"/>
</p>

Then, *FedBed Controller* coordinates the experimentation by, firstly, dividing the parameters into execution, data, and FL learning sub-parameters and invoking the respective subcomponents, namely, Execution, Data, and FL Learning Translator. 
*Execution Translator* takes care of infrastructure-related parameters, generating resource limits by utilizing its resource distributions (e.g., Homogeneous, Gaussian, Pareto, etc.). 
Similarly, *Data Translator* creates the data partitions based on the submitted configurations. 
Lastly, *FL Learning Translator* populates the templates for the FL server and clients, customizing them with the selected ML and aggregation parameters. 
If users would like to introduce custom ML models or aggregation algorithms, they need to materialize the respective interfaces of the FedBed framework.
Specifically, FedBed provides an interface-oriented design that allows users to introduce custom ML models or aggregation strategies in FL services with minimal effort.
With custom artifacts introduced, the FL Learning Translator is responsible to include them in the FL services templates, and, at the runtime, the system invokes these artifacts without needing users to update and rebuild the whole framework.


At the next step, the *Deployment Composer* combines the results from the translators, creating a set of FL deployment objects with all the necessary information and generated parameters. 
To deploy the generated FL workload, FedBed uses a *Virtual Testbed Connector*, which (i) translates the deployment objects into low-level primitives for the underlying Virtual Testbed Orchestrator; (ii) facilitates deployment and testbed configurations; and, at runtime, (iii) retrieves monitoring metrics from the underlying virtualized testbed. 
With the respective primitives on hand, *Virtual Testbed Orchestrator* deploys the FedBed FL services in separate containerized environments, connects them over a virtualized network, and injects the network and computing resource limits.
FedBed integrates two Virtual Testbed Orchestrators, namely, [Fogify](https://github.com/UCY-LINC-LAB/fogify) and [Frisbee](https://github.com/CARV-ICS-FORTH/frisbee). 
These testbed orchestrators are built upon the foundation of multi-host docker orchestrators, such as Swarm and Kubernetes, enabling FedBed to effortlessly scale across an extensive array of nodes.

During experimentation, FedBed gathers various infrastructure and FL service metrics. It retrieves utilization metrics like CPU and memory usage from testbed orchestrators, and, also extracts fine-grained FL metrics from FL Client and Server, including loss, accuracy, and round duration. 
These metrics empower users to evaluate trade-offs between model's performance and the infrastructure's~efficiency.



### Publications

FedBed's paper BibTeX citation:
```
@inproceedings{Symeonides2023,
author    = {Moysis, Symeonides and Fotis, Nikolaidis and Demetris, Trihinas and George, Pallis and Marios D., Dikaiakos and Angelos, Bilas},
title     = {FedBed: Benchmarking Federated Learning over Virtualized Edge Testbeds},
year      = {2023},
publisher = {Association for Computing Machinery},
address   = {New York, NY, USA},
booktitle = {Proceedings of the 16th IEEE/ACM International Conference on Utility and Cloud Computing (UCC 2023)},
location  = {Messina, Italy},
series    = {UCC ’23}
}
```


### Acknowledgements
This work is supported by the Cyprus Research and Innovation Foundation (CODEVELOP-ICT-HEALTH/0322/0047) and the Horizon Europe Framework (dAIEDGE/101120726).

### License
The framework is open-sourced under the Apache 2.0 License base. The codebase of the framework is maintained by the authors for academic research and is therefore provided "as is".
