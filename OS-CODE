import numpy as np
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense


# Step 1: Petri Net Simulation
class PetriNet:
    def __init__(self, places, transitions):
        self.places = places  # list of places
        self.transitions = transitions  # dict mapping transition to input/output places
        self.tokens = defaultdict(int)  # token count in each place

    def add_tokens(self, place, count):
        self.tokens[place] += count

    def fire(self, transition):
        # Check if transition can fire (all input places have enough tokens)
        input_places, output_places = self.transitions[transition]
        if all(self.tokens[place] > 0 for place in input_places):
            for place in input_places:
                self.tokens[place] -= 1  # Consume tokens from input places
            for place in output_places:
                self.tokens[place] += 1  # Add tokens to output places
            return True
        return False  # Transition cannot fire

    def get_state(self):
        return np.array([self.tokens[place] for place in self.places])

    def is_deadlocked(self):
        # If no transitions can fire, we're in a deadlock state
        for transition in self.transitions:
            input_places, _ = self.transitions[transition]
            if all(self.tokens[place] > 0 for place in input_places):
                return False
        return True


# Example Petri net definition: 2 places, 2 transitions
places = ['P1', 'P2']
transitions = {
    'T1': (['P1'], ['P2']),
    'T2': (['P2'], ['P1'])
}

# Initialize Petri net
petri_net = PetriNet(places, transitions)
petri_net.add_tokens('P1', 1)  # Add one token to place P1


# Step 2: Data Generation
def generate_data(petri_net, steps=100):
    data = []
    labels = []
    for _ in range(steps):
        # Record current state
        state = petri_net.get_state()

        # Check if the current state is a deadlock
        deadlock = petri_net.is_deadlocked()

        # Label: 1 for deadlock, 0 for no deadlock
        labels.append(1 if deadlock else 0)

        # Add state to data
        data.append(state)

        if deadlock:
            # Break if deadlocked to avoid infinite loop
            break

        # Randomly fire one of the possible transitions
        for transition in petri_net.transitions:
            if petri_net.fire(transition):
                break  # Fire one transition and move to next step

    return np.array(data), np.array(labels)


# Generate data from the Petri net
data, labels = generate_data(petri_net)
print("Data (state transitions):", data)
print("Labels (deadlock detection):", labels)


# Step 3: Neural Network for Deadlock Prediction
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create and train the model
model = create_model(input_dim=data.shape[1])
model.fit(data, labels, epochs=10, batch_size=1, verbose=1)

# Test with new Petri net states
test_data, _ = generate_data(petri_net, steps=10)
predictions = model.predict(test_data)
print("Predictions (deadlock likelihood):", predictions)
