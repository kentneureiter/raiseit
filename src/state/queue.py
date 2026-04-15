import time

# Maximum people to track in the queue at once
MAX_QUEUE_SIZE = 20


class RaiseQueue:

    def __init__(self):
        # Ordered list of people waiting — first in, first out
        # Each entry: {'person_id': ..., 'time_raised': ..., 'color': ...}
        self.queue = []

    def add(self, person_id):
        
        #Add a person to the queue when their hand raise is confirmed.
        #Ignores them if they're already in the queue.
        
        # Don't add duplicates
        if any(p['person_id'] == person_id for p in self.queue):
            return

        if len(self.queue) < MAX_QUEUE_SIZE:
            self.queue.append({
                'person_id': person_id,
                'time_raised': time.time()
            })

        # Recalculate colors after every change
        self._assign_colors()

    def remove(self, person_id):
        """
        Remove a person from the queue when professor calls on them,
        or when they put their hand down.
        """
        self.queue = [p for p in self.queue if p['person_id'] != person_id]
        self._assign_colors()

    def get_color(self, person_id):
        """
        Returns the color for a given person: 'green', 'yellow', 'red', or None.
        """
        for person in self.queue:
            if person['person_id'] == person_id:
                return person['color']
        return None

    def get_queue(self):
        
        #Returns the full queue in order — useful for driving the light board.
        
        return self.queue

    def clear(self):
        
        #Wipe the whole queue — professor can reset between questions.
        
        self.queue = []

    def _assign_colors(self):
        
        #Private method — recalculates colors based on queue position.
        #1st = green, 2nd = yellow, 3rd and beyond = red.
        
        for i, person in enumerate(self.queue):
            if i == 0:
                person['color'] = 'green'
            elif i == 1:
                person['color'] = 'yellow'
            else:
                person['color'] = 'red'"
