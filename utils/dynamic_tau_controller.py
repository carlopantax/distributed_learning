import numpy as np
from collections import defaultdict, deque
import random
import torch

class DynamicTauController:
    """
    Controller for dynamic adjustment of tau values in local training.
    Tau represents the number of local epochs between synchronizations.
    Tau values are restricted to powers of 2: [4, 8, 16, 32, 64].
    """

    def __init__(self, initial_tau=16, patience=3, improvement_threshold=0.01, stability_window=5):
        """
        Initialize the tau controller.

        Args:
            initial_tau: Starting tau value.
            patience: Number of epochs without improvement before reducing tau.
            improvement_threshold: Minimum improvement in validation accuracy to consider as progress.
            stability_window: Number of recent epochs to track for stability checks.
        """
        self.valid_tau_values = [4, 8, 16, 32, 64]

        
        self.initial_tau = self._get_closest_valid_tau(initial_tau)
        self.min_tau = min(self.valid_tau_values)
        self.max_tau = max(self.valid_tau_values)

        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.stability_window = stability_window

        self.client_states = {}

        self.tau_randomization = True

    def _get_closest_valid_tau(self, tau_value):
        """
        Get the closest valid tau value to the given input.

        Args:
            tau_value: Input tau.

        Returns:
            - Closest valid tau (power of 2).
        """
        return min(self.valid_tau_values, key=lambda x: abs(x - tau_value))

    def _get_next_tau_up(self, current_tau):
        """
        Get the next higher tau value.

        Args:
            current_tau: Current tau.

        Returns:
            - Next higher tau if available, otherwise the same tau.
        """
        current_idx = self.valid_tau_values.index(current_tau)
        if current_idx < len(self.valid_tau_values) - 1:
            return self.valid_tau_values[current_idx + 1]
        return current_tau  

    def _get_next_tau_down(self, current_tau):
        """
        Get the next lower tau value.

        Args:
            current_tau: Current tau.

        Returns:
            - Next lower tau if available, otherwise the same tau.
        """
        current_idx = self.valid_tau_values.index(current_tau)
        if current_idx > 0:
            return self.valid_tau_values[current_idx - 1]
        return current_tau 

    def initialize_client(self, client_id):
        """
        Initialize state tracking for a client.

        Args:
            client_id: Unique identifier of the client.
        """
        base_tau = self.initial_tau

        if self.tau_randomization and len(self.valid_tau_values) > 1:
            current_idx = self.valid_tau_values.index(base_tau)
            possible_indices = [current_idx]

            if current_idx > 0:
                possible_indices.append(current_idx - 1)
            if current_idx < len(self.valid_tau_values) - 1:
                possible_indices.append(current_idx + 1)

            chosen_idx = random.choice(possible_indices)
            initial_tau_with_noise = self.valid_tau_values[chosen_idx]
        else:
            initial_tau_with_noise = base_tau

        self.client_states[client_id] = {
            'current_tau': initial_tau_with_noise,
            'best_val_acc': -float('inf'),
            'best_epoch': 0,
            'epochs_without_improvement': 0,
            'val_acc_history': deque(maxlen=self.stability_window),
            'loss_history': deque(maxlen=self.stability_window),
            'gradient_norm_history': deque(maxlen=self.stability_window),
            'epochs_since_sync': 0,
            'total_epochs': 0,
            'sync_count': 0,
            'tau_change_reason': 'initialized',
            'consecutive_good_epochs': 0,
            'consecutive_bad_epochs': 0,
            'last_sync_performance': 0.0,
            'tau_increase_streak': 0,
            'overfitting_detected': False
        }

    def compute_model_divergence(self,local_model, global_model):
        """
        Compute the L2 norm divergence between local and global model parameters.

        Args:
            local_model: Client's local model.
            global_model: Global reference model.

        Returns:
            - Divergence value as float.
        """
        divergence = 0.0
        with torch.no_grad():
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if local_param.requires_grad:
                    divergence += torch.norm(local_param.data - global_param.data, p=2).item() ** 2

        return divergence ** 0.5 
 

    def should_sync_early(self, client_id, current_val_acc, current_loss, gradient_norm=None, model_divergence=None):
        """
        Decide whether the client should synchronize earlier than its current tau.

        Args:
            client_id: Unique identifier of the client.
            current_val_acc: Current validation accuracy.
            current_loss: Current loss value.
            gradient_norm: Optional gradient norm.
            model_divergence: Optional model divergence.

        Returns:
            - should_sync: Boolean indicating if early synchronization is required.
            - new_tau: Suggested tau value.
        """
        if client_id not in self.client_states:
            self.initialize_client(client_id)

        state = self.client_states[client_id]
        current_local_epoch = state['epochs_since_sync']

        state['val_acc_history'].append(current_val_acc)
        state['loss_history'].append(current_loss)
        if gradient_norm is not None:
            state['gradient_norm_history'].append(gradient_norm)

        if current_local_epoch < self.min_tau:
            return False, state['current_tau']

        improved = current_val_acc > state['best_val_acc'] + self.improvement_threshold

        if improved:
            state['best_val_acc'] = current_val_acc
            state['best_epoch'] = current_local_epoch
            state['epochs_without_improvement'] = 0
            state['consecutive_good_epochs'] += 1
            state['consecutive_bad_epochs'] = 0
        else:
            state['epochs_without_improvement'] += 1
            state['consecutive_bad_epochs'] += 1
            state['consecutive_good_epochs'] = 0

        reached_max_tau = current_local_epoch >= state['current_tau']
        stagnating = state['epochs_without_improvement'] >= self.patience
        diverging = self._detect_divergence(state)
        overfitting = self._detect_overfitting(state)


        state['overfitting_detected'] = overfitting

        should_sync = reached_max_tau or stagnating or diverging or overfitting

        if should_sync:
            new_tau = self._calculate_new_tau(state, current_local_epoch, reached_max_tau,
                                              stagnating, diverging, overfitting, current_val_acc, model_divergence)
        else:
            new_tau = state['current_tau']
            state['tau_change_reason'] = 'continuing'

        return should_sync, new_tau

    def _calculate_new_tau(self, state, current_epoch, reached_max_tau, stagnating,
                       diverging, overfitting, current_val_acc, model_divergence):
        """
        Calculate the new tau value based on training dynamics.

        Args:
            state: Client state dictionary.
            current_epoch: Epoch since last sync.
            reached_max_tau: Whether max tau was reached.
            stagnating: Whether training is stagnating.
            diverging: Whether training is diverging.
            overfitting: Whether overfitting is detected.
            current_val_acc: Current validation accuracy.
            model_divergence: Divergence between local and global models.

        Returns:
            - new_tau: Adjusted tau value.
        """
        current_tau = state['current_tau']
        divergence_threshold = 1.0

        if reached_max_tau and not stagnating and not diverging and not overfitting:
            performance_improvement = current_val_acc - state['last_sync_performance']

            if performance_improvement > self.improvement_threshold * 2 and model_divergence < divergence_threshold:
                # Excellent accuracy and consistency: increase τ by two steps if possible
                new_tau = self._get_next_tau_up(current_tau)
                if new_tau == current_tau and current_tau < self.max_tau:
                    next_up = self._get_next_tau_up(new_tau)
                    if next_up != new_tau:
                        new_tau = next_up
                state['tau_change_reason'] = 'excellent_performance_low_divergence'
                state['tau_increase_streak'] += 1

            elif performance_improvement > self.improvement_threshold or state['consecutive_good_epochs'] >= 3:
                # Good accuracy or positive trend
                new_tau = self._get_next_tau_up(current_tau)
                if performance_improvement > self.improvement_threshold:
                    state['tau_change_reason'] = 'good_performance'
                else:
                    state['tau_change_reason'] = 'consistent_improvement'
                state['tau_increase_streak'] += 1

            else:
                new_tau = current_tau
                state['tau_change_reason'] = 'maintain_tau'
                state['tau_increase_streak'] = 0

        elif stagnating:
            if model_divergence > 1.5 * divergence_threshold:
                # Stagnation and high divergence – reduce τ
                new_tau = self._get_next_tau_down(current_tau)
                state['tau_change_reason'] = 'stagnating_high_divergence'
            elif state['epochs_without_improvement'] > self.patience * 1.5:
                # Severe stagnation – reduce τ by 2 steps if possible
                new_tau = self._get_next_tau_down(current_tau)
                if new_tau != current_tau:
                    next_down = self._get_next_tau_down(new_tau)
                    if next_down != new_tau:
                        new_tau = next_down
                state['tau_change_reason'] = 'severe_stagnation'
            else:
                new_tau = self._get_next_tau_down(current_tau)
                state['tau_change_reason'] = 'mild_stagnation'
            state['tau_increase_streak'] = 0

        elif diverging:
            # Serious divergence – aggressive reduction
            target_tau = max(self.min_tau, min(current_epoch, 8))  
            new_tau = self._get_closest_valid_tau(target_tau)
            state['tau_change_reason'] = 'diverging'
            state['tau_increase_streak'] = 0

        elif overfitting:
            # Overfitting – moderate reduction
            target_tau = min(current_epoch + 1, self._get_next_tau_down(current_tau))
            new_tau = self._get_closest_valid_tau(target_tau)
            state['tau_change_reason'] = 'overfitting'
            state['tau_increase_streak'] = 0

        elif model_divergence > 2 * divergence_threshold:
            # Even without other signals, high divergence – be cautious
            new_tau = self._get_next_tau_down(current_tau)
            state['tau_change_reason'] = 'high_model_divergence'
            state['tau_increase_streak'] = 0

        else:
            new_tau = current_tau
            state['tau_change_reason'] = 'default_maintain'

        return new_tau


    def _detect_divergence(self, state):
        """
        Detect whether training is diverging.

        Args:
            state: Client state.

        Returns:
            - Boolean indicating divergence.
        """
        if len(state['loss_history']) < 3:
            return False

        recent_losses = list(state['loss_history'])[-3:]
        # Check if loss is consistently increasing
        increasing_trend = all(recent_losses[i] <= recent_losses[i + 1] for i in range(len(recent_losses) - 1))

        if len(recent_losses) >= 2:
            sudden_increase = recent_losses[-1] > recent_losses[-2] * 1.2
            return increasing_trend or sudden_increase

        return increasing_trend

    def _detect_overfitting(self, state):
        """
        Detect potential overfitting patterns.

        Args:
            state: Client state.

        Returns:
            - Boolean indicating overfitting.
        """
        if len(state['val_acc_history']) < 3:
            return False

        val_accs = list(state['val_acc_history'])

        # Pattern 1: Validation accuracy decreasing while we're past best epoch
        epochs_past_best = state['epochs_since_sync'] - state['best_epoch']
        recent_decline = (val_accs[-1] < val_accs[-2] and
                          val_accs[-2] < val_accs[-3] and
                          epochs_past_best > 2)

        # Pattern 2: Validation accuracy plateaued well below recent peak
        if len(val_accs) >= self.stability_window:
            recent_std = np.std(val_accs[-3:])
            is_plateaued = recent_std < 0.3
            below_peak = val_accs[-1] < state['best_val_acc'] - self.improvement_threshold * 2

            return recent_decline or (is_plateaued and below_peak and epochs_past_best > self.patience)

        return recent_decline

    def on_sync(self, client_id, new_tau=None):
        """
        Update client state after synchronization with the global model.

        Args:
            client_id: Unique identifier of the client.
            new_tau: Optional new tau value.
        """
        if client_id not in self.client_states:
            self.initialize_client(client_id)

        state = self.client_states[client_id]

        if len(state['val_acc_history']) > 0:
            state['last_sync_performance'] = state['val_acc_history'][-1]

        state['epochs_since_sync'] = 0
        state['sync_count'] += 1

        if new_tau is not None and new_tau != state['current_tau']:
            old_tau = state['current_tau']
            new_tau = self._get_closest_valid_tau(new_tau)
            state['current_tau'] = new_tau
            print(f"Client {client_id}: Tau changed from {old_tau} to {new_tau} ({state['tau_change_reason']})")
        state['best_val_acc'] = -float('inf')
        state['best_epoch'] = 0
        state['epochs_without_improvement'] = 0
        state['consecutive_good_epochs'] = 0
        state['consecutive_bad_epochs'] = 0
        state['overfitting_detected'] = False

    def get_client_tau(self, client_id):
        """
        Get the current tau value for a client.

        Args:
            client_id: Unique identifier of the client.

        Returns:
            - Current tau value.
        """
        if client_id not in self.client_states:
            self.initialize_client(client_id)
        return self.client_states[client_id]['current_tau']

    def get_tau_change_reason(self, client_id):
        """
        Get the reason for the last tau adjustment of a client.

        Args:
            client_id: Unique identifier of the client.

        Returns:
            - Reason string.
        """
        if client_id not in self.client_states:
            return "client_not_initialized"
        return self.client_states[client_id].get('tau_change_reason', 'unknown')

    def step_epoch(self, client_id):
        """
        Increment the epoch counters for a client.

        Args:
            client_id: Unique identifier of the client.
        """
        if client_id not in self.client_states:
            self.initialize_client(client_id)

        state = self.client_states[client_id]
        state['epochs_since_sync'] += 1
        state['total_epochs'] += 1

    def get_client_stats(self, client_id):
        """
        Get a summary of the training statistics for a client.

        Args:
            client_id: Unique identifier of the client.

        Returns:
            - Dictionary with current stats.
        """
        if client_id not in self.client_states:
            return {}

        state = self.client_states[client_id]
        return {
            'current_tau': state['current_tau'],
            'total_epochs': state['total_epochs'],
            'sync_count': state['sync_count'],
            'best_val_acc': state['best_val_acc'],
            'tau_increase_streak': state['tau_increase_streak'],
            'last_reason': state['tau_change_reason'],
            'overfitting_detected': state['overfitting_detected'],
            'valid_tau_values': self.valid_tau_values
        }

    def get_valid_tau_values(self):
        """
        Get the list of valid tau values.

        Returns:
            - Copy of valid tau values.
        """
        return self.valid_tau_values.copy()