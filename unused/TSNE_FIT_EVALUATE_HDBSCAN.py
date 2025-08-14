import os
import numpy as np
import dbcv
from sklearn.cluster import HDBSCAN
import multiprocessing

def cluster_npy_file(args):
    npy_file, minsamples_params = args
    print(f"Clustering {npy_file} with minsamples = {minsamples_params}...")
    # Robust filename parsing with error handling
    try:
        filename = os.path.basename(npy_file)
        parts = filename.split("_")
        perplexity = parts[2].replace("perplexity", "")
        learning_rate = parts[3].replace("learningrate", "").replace(".npy", "")
    except Exception as e:
        print(f"Skipping file {npy_file}: filename parsing error: {e}")
        return []
    tsne = np.load(npy_file)
    results = []
    for minsamples in minsamples_params:
        print(f"Running HDBSCAN with minsamples = {minsamples} on {npy_file}...")
        clustering = HDBSCAN(min_cluster_size=minsamples).fit(tsne)
        labels = clustering.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)
        results.append((perplexity, learning_rate, minsamples, num_clusters, noise_ratio, labels, npy_file))
    return results

if __name__ == "__main__":
    score_log_path = os.path.expandvars("${HOME}/pipeline/esm/hdbscan_cubesearch/hdbscan_dbcv_scores.tsv")
    with open(score_log_path, "w") as f:
        f.write("perplexity\tlearning_rate\tmin_samples\tnum_clusters\tnoise_ratio\tdbcv_score\n")
    npy_dir = os.path.expandvars("${HOME}/pipeline/esm/hdbscan_cubesearch")
    npy_files = [
        os.path.join(npy_dir, f)
        for f in os.listdir(npy_dir)
        if f.startswith("tryp_tsne_perplexity") and f.endswith(".npy")
    ]
    minsamples_params = range(4, 20, 4)
    args_list = [(npy_file, minsamples_params) for npy_file in npy_files]
    all_results = []
    with multiprocessing.Pool(processes=4) as pool:
        for result in pool.map(cluster_npy_file, args_list):
            all_results.extend(result)
    
    # Now run DBCV scoring sequentially in the main process
    for perplexity, learning_rate, minsamples, num_clusters, noise_ratio, labels, npy_file in all_results:
        tsne = np.load(npy_file)
        try:
            tsne_unique, unique_indices = np.unique(tsne, axis=0, return_index=True)
            labels_unique = labels[unique_indices]
            score = dbcv.dbcv(tsne_unique, labels_unique, noise_id=-1)
            print(f"DBCV score for minsamples={minsamples}: {score:.4f}")
        except Exception as e:
            print(f"DBCV computation failed for minsamples={minsamples}: {str(e)}")
            score = None
        with open(score_log_path, "a") as f:
            if score is not None:
                f.write(f"{perplexity}\t{learning_rate}\t{minsamples}\t{num_clusters}\t{noise_ratio:4f}\t{score:.4f}\n")
            else:
                f.write(f"{perplexity}\t{learning_rate}\t{minsamples}\t{num_clusters}\t{noise_ratio:4f}\tNA\n")






