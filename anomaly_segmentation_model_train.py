from utils import MILRankingNetwork, MIL_network_trainer

model = MILRankingNetwork(input_dim=4096).to("cuda")

MIL_network_trainer(model=model,
                    anormal_feat_dir="extracted_anomaly_features",
                    normal_feat_dir="extracted_normal_features",
                    epochs=2000,
                    batch_size=192,
                    learning_rate=0.005)