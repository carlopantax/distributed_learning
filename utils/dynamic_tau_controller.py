import numpy as np
from collections import defaultdict, deque
import random


class DynamicTauController:
    """
    Controls dynamic tau adjustment for each worker based on multiple criteria
    Tau values are restricted to powers of 2: [4, 8, 16, 32, 64]
    """

    def __init__(self, initial_tau=16, patience=3, improvement_threshold=0.01, stability_window=5):
        # Valid tau values (powers of 2)
        self.valid_tau_values = [4, 8, 16, 32, 64]

        # Ensure initial_tau is valid
        self.initial_tau = self._get_closest_valid_tau(initial_tau)
        self.min_tau = min(self.valid_tau_values)
        self.max_tau = max(self.valid_tau_values)

        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.stability_window = stability_window

        # Per-client state
        self.client_states = {}

        # Add some randomization to break synchronization
        self.tau_randomization = True

    def _get_closest_valid_tau(self, tau_value):
        """Get the closest valid tau value (power of 2)"""
        return min(self.valid_tau_values, key=lambda x: abs(x - tau_value))

    def _get_next_tau_up(self, current_tau):
        """Get the next higher valid tau value"""
        current_idx = self.valid_tau_values.index(current_tau)
        if current_idx < len(self.valid_tau_values) - 1:
            return self.valid_tau_values[current_idx + 1]
        return current_tau  # Already at maximum

    def _get_next_tau_down(self, current_tau):
        """Get the next lower valid tau value"""
        current_idx = self.valid_tau_values.index(current_tau)
        if current_idx > 0:
            return self.valid_tau_values[current_idx - 1]
        return current_tau  # Already at minimum

    def initialize_client(self, client_id):
        """Initialize tracking state for a client"""
        # Add slight randomization to initial tau to break synchronization
        base_tau = self.initial_tau

        if self.tau_randomization and len(self.valid_tau_values) > 1:
            # Randomly choose from initial tau or one step up/down
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
            'tau_increase_streak': 0,  # Track consecutive increases
            'overfitting_detected': False
        }

    def should_sync_early(self, client_id, current_val_acc, current_loss, gradient_norm=None):
        """
        Determine if client should sync before reaching current tau
        """
        if client_id not in self.client_states:
            self.initialize_client(client_id)

        state = self.client_states[client_id]
        current_local_epoch = state['epochs_since_sync']

        # Add current metrics to history
        state['val_acc_history'].append(current_val_acc)
        state['loss_history'].append(current_loss)
        if gradient_norm is not None:
            state['gradient_norm_history'].append(gradient_norm)

        # Always complete at least min_tau epochs
        if current_local_epoch < self.min_tau:
            return False, state['current_tau']

        # Check for improvement
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

        # Detect various conditions
        reached_max_tau = current_local_epoch >= state['current_tau']
        stagnating = state['epochs_without_improvement'] >= self.patience
        diverging = self._detect_divergence(state)
        overfitting = self._detect_overfitting(state)

        # Update overfitting detection
        state['overfitting_detected'] = overfitting

        should_sync = reached_max_tau or stagnating or diverging or overfitting

        if should_sync:
            new_tau = self._calculate_new_tau(state, current_local_epoch, reached_max_tau,
                                              stagnating, diverging, overfitting, current_val_acc)
        else:
            new_tau = state['current_tau']
            state['tau_change_reason'] = 'continuing'

        return should_sync, new_tau

    def _calculate_new_tau(self, state, current_epoch, reached_max_tau, stagnating,
                           diverging, overfitting, current_val_acc):
        """Calculate new tau based on training conditions - restricted to power-of-2 values"""
        current_tau = state['current_tau']

        if reached_max_tau and not stagnating and not diverging and not overfitting:
            # Completed full tau successfully - consider increasing
            performance_improvement = current_val_acc - state['last_sync_performance']

            if performance_improvement > self.improvement_threshold * 2:
                # Excellent performance - increase tau by 2 steps if possible
                new_tau = self._get_next_tau_up(current_tau)
                if new_tau == current_tau and current_tau < self.max_tau:
                    # Try to go up by 2 levels if we're not at max
                    next_up = self._get_next_tau_up(new_tau)
                    if next_up != new_tau:
                        new_tau = next_up

                state['tau_change_reason'] = 'excellent_performance'
                state['tau_increase_streak'] += 1

            elif (performance_improvement > self.improvement_threshold or
                  state['consecutive_good_epochs'] >= 3):
                # Good performance or consistent improvement - increase by 1 step
                new_tau = self._get_next_tau_up(current_tau)

                if performance_improvement > self.improvement_threshold:
                    state['tau_change_reason'] = 'good_performance'
                    state['tau_increase_streak'] += 1
                else:
                    state['tau_change_reason'] = 'consistent_improvement'
                    state['tau_increase_streak'] = max(0, state['tau_increase_streak'])
            else:
                # Maintain current tau
                new_tau = current_tau
                state['tau_change_reason'] = 'maintain_tau'
                state['tau_increase_streak'] = 0

        elif stagnating:
            # No improvement for several epochs - decrease tau
            if state['epochs_without_improvement'] > self.patience * 1.5:
                # Severe stagnation - reduce tau by 2 steps if possible
                new_tau = self._get_next_tau_down(current_tau)
                if new_tau != current_tau:  # Successfully went down one step
                    next_down = self._get_next_tau_down(new_tau)
                    if next_down != new_tau:  # Can go down another step
                        new_tau = next_down

                state['tau_change_reason'] = 'severe_stagnation'
            else:
                # Mild stagnation - reduce tau by 1 step
                new_tau = self._get_next_tau_down(current_tau)
                state['tau_change_reason'] = 'mild_stagnation'

            state['tau_increase_streak'] = 0

        elif diverging:
            # Training is diverging - aggressive tau reduction
            # Go to minimum tau or current epoch count (whichever is larger but valid)
            target_tau = max(self.min_tau, min(current_epoch, 8))  # Cap at 8 for diverging
            new_tau = self._get_closest_valid_tau(target_tau)
            state['tau_change_reason'] = 'diverging'
            state['tau_increase_streak'] = 0

        elif overfitting:
            # Overfitting detected - moderate reduction
            # Reduce by 1 step or go to a reasonable tau based on current epoch
            target_tau = min(current_epoch + 1, self._get_next_tau_down(current_tau))
            new_tau = self._get_closest_valid_tau(target_tau)
            state['tau_change_reason'] = 'overfitting'
            state['tau_increase_streak'] = 0

        else:
            # Default case - maintain tau
            new_tau = current_tau
            state['tau_change_reason'] = 'default_maintain'

        return new_tau

    def _detect_divergence(self, state):
        """Detect if training is diverging (loss increasing consistently)"""
        if len(state['loss_history']) < 3:
            return False

        recent_losses = list(state['loss_history'])[-3:]
        # Check if loss is consistently increasing
        increasing_trend = all(recent_losses[i] <= recent_losses[i + 1] for i in range(len(recent_losses) - 1))

        # Also check for sudden large increase
        if len(recent_losses) >= 2:
            sudden_increase = recent_losses[-1] > recent_losses[-2] * 1.2
            return increasing_trend or sudden_increase

        return increasing_trend

    def _detect_overfitting(self, state):
        """Detect potential overfitting patterns"""
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
        """Called when client synchronizes with global model"""
        if client_id not in self.client_states:
            self.initialize_client(client_id)

        state = self.client_states[client_id]

        # Store performance for next comparison
        if len(state['val_acc_history']) > 0:
            state['last_sync_performance'] = state['val_acc_history'][-1]

        state['epochs_since_sync'] = 0
        state['sync_count'] += 1

        if new_tau is not None and new_tau != state['current_tau']:
            old_tau = state['current_tau']
            # Ensure new_tau is valid
            new_tau = self._get_closest_valid_tau(new_tau)
            state['current_tau'] = new_tau
            print(f"Client {client_id}: Tau changed from {old_tau} to {new_tau} ({state['tau_change_reason']})")

        # Reset tracking state but keep some momentum
        state['best_val_acc'] = -float('inf')
        state['best_epoch'] = 0
        state['epochs_without_improvement'] = 0
        state['consecutive_good_epochs'] = 0
        state['consecutive_bad_epochs'] = 0
        state['overfitting_detected'] = False

    def get_client_tau(self, client_id):
        """Get current tau for a client"""
        if client_id not in self.client_states:
            self.initialize_client(client_id)
        return self.client_states[client_id]['current_tau']

    def get_tau_change_reason(self, client_id):
        """Get the reason for the last tau change"""
        if client_id not in self.client_states:
            return "client_not_initialized"
        return self.client_states[client_id].get('tau_change_reason', 'unknown')

    def step_epoch(self, client_id):
        """Increment epoch counters for a client"""
        if client_id not in self.client_states:
            self.initialize_client(client_id)

        state = self.client_states[client_id]
        state['epochs_since_sync'] += 1
        state['total_epochs'] += 1

    def get_client_stats(self, client_id):
        """Get comprehensive stats for a client"""
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
        """Get list of valid tau values"""
        return self.valid_tau_values.copy()