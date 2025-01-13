from sklearn.metrics.cluster import rand_score, adjusted_rand_score


def calculate_rand(dgd_rep, data_loader):

   labels = data_loader.dataset.label

   clustering = []

   

   df = pd.DataFrame(labels, columns=["label"])

   

   rep = dgd_rep

   for i in range(len(rep.z)):

       cluster = dgd.gmm.clustering(rep(i)).unsqueeze(0).detach().cpu().numpy()

       clustering.extend(cluster)

   df["cluster"] = clustering

   df["cluster"] = df["cluster"].astype('category')


   label_counts = df.groupby(['cluster', 'label'], observed=False).size().reset_index(name='counts')

   most_frequent_labels = label_counts.loc[label_counts.groupby('cluster', observed=False)['counts'].idxmax()]

   cluster_to_label = dict(zip(most_frequent_labels['cluster'], most_frequent_labels['label']))

   df['cluster_name'] = df['cluster'].map(cluster_to_label)


   rand_index = adjusted_rand_score(df["label"], df["cluster_name"])


   return rand_index

usage:

print(calculate_rand(dgd.train_rep, train_loader))