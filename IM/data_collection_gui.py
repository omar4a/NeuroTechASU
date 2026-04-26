import customtkinter as ctk
import random
from pylsl import StreamInfo, StreamOutlet, local_clock

# Configure the UI framework
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DataCollectionGUI(ctk.CTk):
    """
    data_collection_gui.py (The Modified PhysioNet Paradigm)
    A GUI to guide the user through motor imagery training and record timestamped markers.
    """
    def __init__(self):
        super().__init__()

        self.title("Motor Imagery Data Collection")
        self.geometry("800x600")

        # Initialize LSL StreamOutlet for string markers
        info = StreamInfo('IM_Markers', 'Markers', 1, 0, 'string', 'im_markers_123')
        self.outlet = StreamOutlet(info)

        # Central label for instructions and cues
        self.label = ctk.CTkLabel(self, text="Press Start to Begin", font=("Inter", 48))
        self.label.place(relx=0.5, rely=0.5, anchor="center")

        # Start Button
        self.start_button = ctk.CTkButton(self, text="Start Protocol", command=self.start_protocol, font=("Inter", 24))
        self.start_button.place(relx=0.5, rely=0.8, anchor="center")

        # Define the trial types
        self.trials = ["Left_MI", "Right_MI", "Rest"]
        self.is_running = False

    def start_protocol(self):
        self.start_button.destroy()
        self.is_running = True
        self.next_trial()

    def next_trial(self):
        if not self.is_running:
            return
            
        # Step 1: Blank screen for 2.0 seconds (relaxation)
        self.label.configure(text="", text_color="white")
        self.update()
        
        # Schedule the cue display
        self.after(2000, self.show_cue)

    def show_cue(self):
        if not self.is_running:
            return
            
        # Select random trial type
        trial_type = random.choice(self.trials)
        
        # Configure Display Cue and Marker based on trial type
        if trial_type == "Left_MI":
            self.label.configure(text="← Left Hand", text_color="cyan")
            marker = "Left_MI_Start"
        elif trial_type == "Right_MI":
            self.label.configure(text="Right Hand →", text_color="cyan")
            marker = "Right_MI_Start"
        else:
            self.label.configure(text="↓ Rest", text_color="gray")
            marker = "Rest_Start"
            
        self.update()
        
        # Simultaneously push the LSL marker with a precise timestamp
        self.outlet.push_sample([marker], local_clock())
        
        # Step 2: Hold the cue on screen for exactly 4.0 seconds
        self.after(4000, self.remove_cue)

    def remove_cue(self):
        if not self.is_running:
            return
            
        # Optional: Display a fixation cross during the ITI
        self.label.configure(text="+", text_color="white")
        self.update()
        
        # Step 3: Randomized inter-trial interval between 1.5 and 2.5 seconds
        iti = random.uniform(1.5, 2.5)
        self.after(int(iti * 1000), self.next_trial)

if __name__ == "__main__":
    app = DataCollectionGUI()
    app.mainloop()
