import customtkinter as ctk
import subprocess
import threading
import sys
import os

# Configure the UI framework
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class MainDashboard(ctk.CTk):
    """
    main_dashboard.py (The Integration)
    A dashboard to launch the background realtime inference process and
    visually display the debounced classification output for the software team.
    """
    def __init__(self):
        super().__init__()

        self.title("BCI Integration Dashboard")
        self.geometry("900x500")

        # Title
        self.title_label = ctk.CTkLabel(self, text="Real-Time Decoder Status", font=("Inter", 32, "bold"))
        self.title_label.pack(pady=40)

        # Large Text Display for Classification Output
        self.state_label = ctk.CTkLabel(self, text="WAITING FOR ENGINE...", font=("Inter", 64, "bold"), text_color="yellow")
        self.state_label.pack(pady=40, expand=True)

        # Launch Button
        self.start_btn = ctk.CTkButton(self, text="Launch Inference Engine", command=self.launch_engine, font=("Inter", 20), height=50)
        self.start_btn.pack(pady=30)
        
        self.process = None

    def launch_engine(self):
        # Disable button after launching
        self.start_btn.configure(state="disabled", text="Engine Running")
        
        # Path to inference script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_inference.py")
        
        # Launch background process
        self.process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1 # Line buffered
        )
        
        # Start a thread to read stdout without blocking the GUI
        self.read_thread = threading.Thread(target=self.read_output, daemon=True)
        self.read_thread.start()

    def read_output(self):
        # Continually read output from the inference engine
        for line in iter(self.process.stdout.readline, ''):
            line = line.strip()
            
            # Check for our specific STATE output format
            if line.startswith("STATE:"):
                state = line.split("STATE:")[1].strip()
                # Safely update GUI from the thread using .after()
                self.after(0, self.update_state_label, state)
            else:
                # Print debug output to standard console
                print(f"[Engine] {line}")
                
    def update_state_label(self, state):
        # Dynamically change color based on predicted state
        colors = {
            "RESTING": "gray",
            "OPENING FIST": "cyan",
            "CLOSING FIST": "orange"
        }
        color = colors.get(state, "white")
        self.state_label.configure(text=state, text_color=color)

    def on_closing(self):
        # Cleanly terminate the subprocess when window is closed
        if self.process:
            self.process.terminate()
        self.destroy()

if __name__ == "__main__":
    app = MainDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
