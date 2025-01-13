import torch
from tqdm import tqdm
from src.dgd.latent import RepresentationLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_potential_reps(sample_list):
    """
    takes a list of samples drawn from the DGD's distributions.
    The length gives the number of distributions which defines
    the dimensionality of the output tensor.
    If the list of samples is longer than 1, we will create representations
    from the combination of each GMM's samples.
    """
    return sample_list[0]
    
def learn_new_representation(dgd, 
                             data_loader,
                             test_epochs=50,
                             learning_rates=1e-2, 
                             weight_decay=0.,
                             betas=(0.5, 0.7),
                             reduction_type="sum",
                             resampling_type="mean"):
    """
    This function learns a new representation layer for the DGD.
    The new representation layer is learned by sampling new points
    from the GMMs and finding the best fitting GMM for each sample.
    The new representation layer is then optimized to minimize the
    reconstruction loss of the DGD.
    """
    # test_data = torch.Tensor(data_loader.dataset.data)
    # test_scaling_factors = torch.mean(test_data, dim=-1).unsqueeze(1).to(device)  # Scale each sample

    gmm_loss = True
    n_samples_new = len(data_loader.dataset)
    potential_reps = prepare_potential_reps([dgd.gmm.sample_new_points(resampling_type)]) # initialize reps

    dgd.eval() # evaluation mode
    X_test = dgd.decoder(potential_reps.to(device)) # reconstructed data

    rep_init_values = torch.zeros((n_samples_new, potential_reps.shape[-1]))

    for (mut_data, lib, i) in data_loader: # loop through samples
        loss = torch.empty(0).to(device) # placed on the same hardware for computations
        for X in X_test:   # loop through each empty reconstruction
            mut_recon_loss = dgd.decoder.loss( # 
                nn_output=X.to(device), 
                target=mut_data.to(device), 
                scale=lib,
                feature_ids=None, 
                reduction="sum"
            )
            loss = torch.cat((loss, mut_recon_loss.unsqueeze(0)))
        best_fit_ids = torch.argmin(loss, dim=-1).detach().cpu()
        rep_init_values[i, :] = potential_reps.clone()[best_fit_ids, :]

    Ntest=len(data_loader.dataset)
    new_rep = RepresentationLayer(n_rep=dgd.rep_dim, # set-up the representation layer
                                  n_sample=Ntest,
                                  value_init=rep_init_values).to(device)
    
    test_rep_optimizer = torch.optim.AdamW(new_rep.parameters(), lr=learning_rates, weight_decay=weight_decay, betas=betas) # set-up optimizer
    
    for epoch in tqdm(range(test_epochs)): # loop through epochs for a progress bar #tqdm()
        test_rep_optimizer.zero_grad()
        for (mut_data, lib, index) in data_loader: # loop through samples
            mut_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=new_rep(index),
                target=[mut_data.to(device)],
                scale=lib.unsqueeze(1).to(device), # dimensional alignment with unsqueeze
                gmm_loss=gmm_loss,
                reduction=reduction_type
            )
            loss = mut_recon_loss + gmm_loss
            loss.backward()
        test_rep_optimizer.step() # updating the rep for each epoch
    
    return new_rep 