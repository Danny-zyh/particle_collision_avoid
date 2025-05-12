import numpy as np

class ConformalRegion:
    def __init__(
        self, 
        traj_path,
        pred_path, 
        pred_horizon,
        failure_rate,
        single_agent=True
    ):
        '''
        Args:
            traj_path (str): Path to the trajectory data.
            pred_path (str): Path to the prediction data.
            pred_horizon (int): The prediction horizon into the future.
            failure_rate (float): The acceptable failure rate for the prediction regions.
        '''
        self.traj = np.load(traj_path, allow_pickle=True)
        self.pred = np.load(pred_path, allow_pickle=True)
        self.pred_horizon = pred_horizon
        self.failure_rate = failure_rate
        self.single_agent = single_agent
        if single_agent:
            self.n_traj, self.traj_len, self.state_dim = self.traj.shape
        else:
            self.n_agents, self.n_traj, self.traj_len, self.state_dim = self.traj.shape

    def get_single_agent_nonconformity_scores(self, t, tau):
        """
        For each trajectory, the nonconformity score for prediction tau at time t is
        the distance between the predicted state and the actual state of the trajectory at time t+tau.
        
        Args:
            t (int): The time step at which to compute the nonconformity score.
            tau (int): The prediction horizon ahead of t.
            
        Returns:
            np.ndarray: The nonconformity scores for each trajectory.
        """
        
        assert self.single_agent, "This method is for single agent only"
        assert tau < self.pred_horizon, "tau must be less than pred_horizon"
        assert t + tau < self.traj_len, "t + tau must be less than traj_len"
        
        pred_state = self.pred[:, t, tau, :]
        actual_state = self.traj[:, t + tau, :]
        nonconformity_scores = np.linalg.norm(pred_state - actual_state, axis=1)
        
        return nonconformity_scores
    
    def get_multi_agent_nonconformity_scores(self, t, tau):
        """
        For each trajectory, the multi-agent nonconformity score for prediction tau at time t is the maximum distance
        between the predicted state and the actual state of the trajectory at time t+tau across all agents.
        
        Args:
            t (int): The time step at which to compute the nonconformity score.
            tau (int): The prediction horizon ahead of t.
            
        Returns:
            np.ndarray: The nonconformity scores for each trajectory.
        """
        
        assert not self.single_agent, "This method is for multi-agent only"
        assert tau < self.pred_horizon, "tau must be less than pred_horizon"
        assert t + tau < self.traj_len, "t + tau must be less than traj_len"
        
        pred_state = self.pred[:, :, t, tau, :]
        actual_state = self.traj[:, :, t + tau, :]
        nonconformity_scores = np.linalg.norm(pred_state - actual_state, axis=2)
        nonconformity_scores = np.max(nonconformity_scores, axis=1)
        
        return nonconformity_scores
    
    def construct_valid_prediction_regions(self, t, tau):
        """
        Constructs valid prediction regions for the trajectories at time t and prediction horizon tau.
        
        Args:
            t (int): The time step at which to compute the valid prediction regions.
            tau (int): The prediction horizon ahead of t.
            alpha (float): The failure rate for the prediction regions.
            
        Returns:
            float: The valid prediction region for (t, tau).
        """
        # Get the nonconformity scores for the trajectories
        nonconformity_scores = self.get_single_agent_nonconformity_scores(t, tau)
        # Sort the nonconformity scores
        sorted_scores = np.sort(nonconformity_scores)
        # Add infinity to the end of the sorted scores
        sorted_scores = np.append(sorted_scores, np.inf)
        # Compute the index for the valid prediction region
        index = int(np.floor((1 - self.failure_rate) * (self.n_traj + 1)))
        # Get the valid prediction region
        valid_region = sorted_scores[index]
        
        return valid_region

if __name__ == "__main__":
    traj_path = 'data/random_5_p3_p1.npy'
    pred_path = 'data/random_5_p3_p1_pred_20.npy'
    pred_horizon = 20
    failure_rate = 0.1
    conformal_region = ConformalRegion(traj_path, pred_path, pred_horizon, failure_rate)
    t = 10
    tau = 5
    nonconformity_scores = conformal_region.get_single_agent_nonconformity_scores(t, tau)
    print("Nonconformity scores shape:", nonconformity_scores.shape)
    valid_region = conformal_region.construct_valid_prediction_regions(t, tau)
    print("Valid prediction region:", valid_region)
    
    

