Paper of the source codes released:  
Tian Bian, Xi Xiao, Tingyang Xu, Peilin Zhao, Wenbing Huang, Yu Rong, Junzhou Huang. Rumor Detectionon Social Media with Bi-Directional Graph Convolutional Networks. AAAI 2020.

Datasets:  
The datasets used in the experiments were based on the three publicly available Weibo and Twitter datasets released by Ma et al. (2016) and Ma et al. (2017):

Jing Ma, Wei Gao, Prasenjit Mitra, Sejeong Kwon, Bernard J Jansen, Kam-Fai Wong, and Meeyoung Cha. Detecting rumors from microblogs with recurrent neural networks. In Proceedings of IJCAI 2016.

Jing Ma, Wei Gao, Kam-Fai Wong. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning. ACL 2017.

In the 'data' folder we provide the pre-processed data files used for our experiments. The raw datasets can be respectively downloaded from https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0. and https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0. For details about the datasets please contact Jing at: majing at se dot cuhk dot edu dot hk.

The Weibo datafile 'weibotree.txt' is in a tab-sepreted column format, where each row corresponds to a tweet. Consecutive columns correspond to the following pieces of information:  
1: root-id -- an unique identifier describing the tree (weiboid of the root);  
2: index-of-parent-weibo -- an index number of the parent weibo for the current weibo;  
3: index-of-the-current-weibo -- an index number of the current weibo;  
4: Fill with '1', no special meaning.  
5: Fill with '1', no special meaning.  
6: list-of-index-and-counts -- the rest of the line contains space separated index-count pairs, where a index-count pair is in format of "index:count", E.g., "index1:count1 index2:count2" (extracted from the "text" field in the json format from Weibo raw datasets)  

The Twitter datafile 'data.TD_RvNN.vol_5000.txt' is in a tab-sepreted column format, where each row corresponds to a tweet. Consecutive columns correspond to the following pieces of information:  
1: root-id -- an unique identifier describing the tree (tweetid of the root);  
2: index-of-parent-tweet -- an index number of the parent tweet for the current tweet;  
3: index-of-the-current-tweet -- an index number of the current tweet;  
4: parent-number -- the total number of the parent node in the tree that the current tweet is belong to;  
5: text-length -- the maximum length of all the texts from the tree that the current tweet is belong to;  
6: list-of-index-and-counts -- the rest of the line contains space separated index-count pairs, where a index-count pair is in format of "index:count", E.g., "index1:count1 index2:count2" (extracted from the "text" field in the json format from Twitter)  

Dependencies:  
python==3.5.2  
numpy==1.18.1  
torch==1.4.0  
torch_scatter==1.4.0  
torch_sparse==0.4.3  
torch_cluster==1.4.5  
torch_geometric==1.3.2  
tqdm==4.40.0  
joblib==0.14.1  

Reproduce the experimental results:  
Run script "main.sh", choose "model/Weibo/BiGCN_Weibo.py" for BiGCN model on Weibo dataset or "model/Titter/BiGCN_Twitter.py" on Twitter15/Twitter16 dataset with
args: DatasetName, Iterations; E.g., run 'python ./model/Twitter/BiGCN_Twitter.py Twitter15 100' in "main.sh" to reproduce the experimental results of 100 iterations of BiGCN model on Twitter15 dataset.  
 
If you find this code useful, please let us know and cite our paper.  
If you have any question, please contact Tian at: bt18 at mails dot tsinghua dot edu dot cn.