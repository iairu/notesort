import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, ttk
import json
import os
import subprocess
import sys
import venv
import shutil
from tqdm import tqdm
import webbrowser
import threading
import time

class NoteSort:
    def __init__(self, master):
        self.master = master
        self.master.title("NoteSort")
        self.master.geometry("800x600")

        self.setup_environment()
        self.create_gui()

    def setup_environment(self):
        venv_dir = 'venv'
        if not os.path.exists(venv_dir):
            print("Creating virtual environment...")
            venv.create(venv_dir, with_pip=True)

        # Check if packages are already installed
        if os.path.exists('installed.tmp'):
            print("Packages already installed. Skipping installation.")
            return

        # Install required packages from requirements.txt
        requirements_file = 'requirements.txt'
        if os.path.exists(requirements_file):
            with open(requirements_file, 'r') as f:
                packages = f.read().splitlines()
            
            progress_window = tk.Toplevel(self.master)
            progress_window.title("Setting up environment")
            progress_label = tk.Label(progress_window, text="Checking packages...")
            progress_label.pack(pady=10)
            progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress_bar.pack(pady=10)
            
            packages_to_install = []
            for i, package in enumerate(tqdm(packages, desc="Checking packages")):
                try:
                    __import__(package.split('==')[0])
                except ImportError:
                    packages_to_install.append(package)
                progress_bar['value'] = (i + 1) / len(packages) * 100
                progress_window.update()
            
            if packages_to_install:
                progress_label.config(text="Installing packages...")
                progress_bar['value'] = 0
                for i, package in enumerate(tqdm(packages_to_install, desc="Installing packages")):
                    print(f"Installing {package}...")
                    subprocess.check_call([os.path.join(venv_dir, 'bin', 'pip'), "install", package])
                    progress_bar['value'] = (i + 1) / len(packages_to_install) * 100
                    progress_window.update()
                
                # Create installed.tmp file after successful installation
                with open('installed.tmp', 'w') as f:
                    f.write('Packages installed successfully')
            else:
                print("All required packages are already installed.")
            
            progress_window.destroy()
        else:
            print("requirements.txt not found. Please ensure it exists in the project directory.")

    def create_gui(self):
        # Create menubar
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        # Create Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Readme", command=self.show_help)

        # Create three frames for each stage
        self.training_frame = tk.LabelFrame(self.master, text="Training", padx=10, pady=10)
        self.training_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.inferring_frame = tk.LabelFrame(self.master, text="Inferring", padx=10, pady=10)
        self.inferring_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.validation_frame = tk.LabelFrame(self.master, text="Validation", padx=10, pady=10)
        self.validation_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)
        self.master.grid_rowconfigure(0, weight=1)

        # Set smaller font size
        small_font = ('TkDefaultFont', 8)

        # Training frame contents
        training_top_frame = tk.Frame(self.training_frame)
        training_top_frame.pack(side=tk.TOP, fill=tk.X)

        # Labels file section
        labels_frame = tk.LabelFrame(training_top_frame, text="Labels", padx=5, pady=5)
        labels_frame.pack(fill=tk.X, pady=5)

        self.choose_labels_button = tk.Button(labels_frame, text="Choose Labels File (Optional)",
                                              command=self.choose_labels_file, font=small_font)
        self.choose_labels_button.pack(pady=2)
        self.generate_labels_button = tk.Button(labels_frame, text="Add or Edit Labels",
                                                command=self.generate_labels, font=small_font)
        self.generate_labels_button.pack(pady=2)

        # Check if train_labels.json exists
        self.train_labels_exists = False
        self.train_labels_status = tk.Label(labels_frame, text="train_labels.json not found",
                                            fg="red", font=small_font)
        self.train_labels_status.pack(pady=2)
        self.open_train_labels_button = tk.Button(labels_frame, text="(Open train_labels.json)", 
                                                  command=lambda: self.open_file('train_labels.json'),
                                                  font=small_font)

        # Training input section
        input_frame = tk.LabelFrame(training_top_frame, text="Dataset", padx=5, pady=5)
        input_frame.pack(fill=tk.X, pady=5)

        tk.Button(input_frame, text="Assign Labels to Paragraphs in Browser",
                  command=self.assign_buckets, wraplength=150, font=small_font).pack(pady=2)
        tk.Label(input_frame,
                 text="Use labeling.html to load train_labels.json and sort paragraphs into buckets",
                 wraplength=200, font=small_font).pack(pady=2)

        tk.Button(input_frame, text="Choose train_input.json", command=self.choose_train_input,
                  font=small_font).pack(pady=2)

        # Check if train_input.json exists
        self.train_input_exists = False
        self.train_input_status = tk.Label(input_frame, text="train_input.json not found",
                                           fg="red", font=small_font)
        self.train_input_status.pack(pady=2)
        self.open_train_input_button = tk.Button(input_frame, text="(Open train_input.json)", 
                                                 command=lambda: self.open_file('train_input.json'),
                                                 font=small_font)

        # Training process section
        training_process_frame = tk.LabelFrame(training_top_frame, text="Process", padx=5, pady=5)
        training_process_frame.pack(fill=tk.X, pady=5)

        self.start_training_button = tk.Button(training_process_frame, text="Start Training",
                  command=self.start_training, font=small_font, state=tk.DISABLED)
        self.start_training_button.pack(pady=2)
        tk.Label(training_process_frame,
                 text="Run distilbert_train.py to train the model on train_input.json",
                 wraplength=200, font=small_font).pack(pady=2)

        # Inferring frame contents
        inferring_dataset_frame = tk.LabelFrame(self.inferring_frame, text="Dataset", padx=5, pady=5)
        inferring_dataset_frame.pack(fill=tk.X, pady=5)

        self.choose_input_button = tk.Button(inferring_dataset_frame, text="Choose Input MD File (Optional)",
                                             command=self.choose_input_file, font=small_font)
        self.choose_input_button.pack(pady=2)
        self.create_input_button = tk.Button(inferring_dataset_frame, text="Create or Edit Input MD File",
                                             command=self.create_input_file, font=small_font)
        self.create_input_button.pack(pady=2)

        # Check if infer_input.md exists
        self.infer_input_exists = False
        self.infer_input_status = tk.Label(inferring_dataset_frame, text="infer_input.md not found",
                                           fg="red", font=small_font)
        self.infer_input_status.pack(pady=2)
        self.open_infer_input_button = tk.Button(inferring_dataset_frame, text="(Open infer_input.md)",
                                                 command=lambda: self.open_file('infer_input.md'), 
                                                 font=small_font)

        # Inferring process frame
        inferring_process_frame = tk.LabelFrame(self.inferring_frame, text="Process", padx=5, pady=5)
        inferring_process_frame.pack(fill=tk.X, pady=5)
        
        # Checkbox for top label only
        self.top_label_only = tk.BooleanVar()
        tk.Checkbutton(inferring_process_frame, text="Only keep top label in output",
                       variable=self.top_label_only, font=small_font).pack(pady=2)
        
        self.start_inferring_button = tk.Button(inferring_process_frame, text="Start Inferring",
                  command=self.start_inferring, font=small_font, state=tk.DISABLED)
        self.start_inferring_button.pack(pady=2)
        tk.Label(inferring_process_frame, text="Run distilbert_infer.py to execute the model",
                 wraplength=200, font=small_font).pack(pady=2)

        # Check if infer_output.json exists
        self.infer_output_exists = False
        self.open_infer_output_button = tk.Button(inferring_process_frame, text="(Open infer_output.json)",
                                                  command=lambda: self.open_file('infer_output.json'), 
                                                  font=small_font)

        # Validation frame contents
        validation_process_frame = tk.LabelFrame(self.validation_frame, text="Process", padx=5, pady=5)
        validation_process_frame.pack(fill=tk.X, pady=5)

        tk.Button(validation_process_frame, text="Start Validation",
                  command=self.start_validation, font=small_font).pack(pady=5)
        tk.Label(validation_process_frame, text="Run distilbert_validate.py to perform validation",
                 wraplength=200, font=small_font).pack(pady=5)

        validation_results_frame = tk.LabelFrame(self.validation_frame, text="Results", padx=5, pady=5)
        validation_results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.validation_output = tk.Text(validation_results_frame, wrap=tk.WORD, height=20, width=40, font=small_font)
        self.validation_output.pack(pady=5, expand=True, fill=tk.BOTH)

        validation_saving_frame = tk.LabelFrame(self.validation_frame, text="Saving", padx=5, pady=5)
        validation_saving_frame.pack(fill=tk.X, pady=5)

        self.save_validation_button = tk.Button(validation_saving_frame, text="Save Validation Output",
                  command=self.save_validation_output, font=small_font, state=tk.DISABLED)
        self.save_validation_button.pack(pady=5)

        # Check if validation_output.txt exists
        self.validation_output_exists = False
        self.open_validation_output_button = tk.Button(validation_saving_frame, text="(Open validation_output.txt)",
                                                       command=lambda: self.open_file('validation_output.txt'), 
                                                       font=small_font)
        
        # Bind text modification event to enable/disable save button
        self.validation_output.bind('<<Modified>>', self.check_validation_output)

        # Start the file checking interval
        self.check_files_and_update_buttons()
        self.master.after(2000, self.check_files_and_update_buttons)

    def check_files_and_update_buttons(self):
        self.train_labels_exists = os.path.exists('train_labels.json')
        self.train_input_exists = os.path.exists('train_input.json')
        self.infer_input_exists = os.path.exists('infer_input.md')
        self.infer_output_exists = os.path.exists('infer_output.json')
        self.validation_output_exists = os.path.exists('validation_output.txt')
        self.trained_model_exists = os.path.isdir('trained_model_1')

        if self.train_labels_exists:
            self.train_labels_status.config(text="train_labels.json detected", fg="green")
            self.open_train_labels_button.pack(pady=2)
        else:
            self.train_labels_status.config(text="train_labels.json not found", fg="red")
            self.open_train_labels_button.pack_forget()

        if self.train_input_exists:
            self.train_input_status.config(text="train_input.json detected", fg="green")
            self.open_train_input_button.pack(pady=2)
        else:
            self.train_input_status.config(text="train_input.json not found", fg="red")
            self.open_train_input_button.pack_forget()

        if self.infer_input_exists:
            self.infer_input_status.config(text="infer_input.md detected", fg="green")
            self.open_infer_input_button.pack(pady=5)
        else:
            self.infer_input_status.config(text="infer_input.md not found", fg="red")
            self.open_infer_input_button.pack_forget()

        if self.infer_output_exists:
            self.open_infer_output_button.pack(pady=5)
        else:
            self.open_infer_output_button.pack_forget()

        if self.validation_output_exists:
            self.open_validation_output_button.pack(pady=5)
        else:
            self.open_validation_output_button.pack_forget()

        # Update Start Training button state
        if self.train_input_exists and self.train_labels_exists:
            self.start_training_button.config(state=tk.NORMAL)
        else:
            self.start_training_button.config(state=tk.DISABLED)

        # Update Start Inferring button state
        if self.infer_input_exists and self.trained_model_exists:
            self.start_inferring_button.config(state=tk.NORMAL)
        else:
            self.start_inferring_button.config(state=tk.DISABLED)

        self.master.after(2000, self.check_files_and_update_buttons)

    def check_validation_output(self, event):
        self.save_validation_button.config(state=tk.NORMAL if self.validation_output.edit_modified() else tk.DISABLED)

    def open_file(self, file_path):
        if os.path.exists(file_path):
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.call(["open", file_path])
            else:
                subprocess.call(["xdg-open", file_path])
        else:
            messagebox.showerror("Error", f"{file_path} not found.")

    def choose_train_input(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            if os.path.exists('train_input.json'):
                if messagebox.askyesno("File Exists", "train_input.json already exists. Do you want to replace it?"):
                    self.copy_file(file_path, 'train_input.json')
            else:
                self.copy_file(file_path, 'train_input.json')

    def save_validation_output(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.validation_output.get('1.0', tk.END))

    def update_labels_buttons(self):
        if self.labels_var.get() == "choose":
            self.choose_labels_button.config(state=tk.NORMAL)
            self.generate_labels_button.config(state=tk.DISABLED)
        else:
            self.choose_labels_button.config(state=tk.DISABLED)
            self.generate_labels_button.config(state=tk.NORMAL)

    def update_input_buttons(self):
        if self.input_var.get() == "choose":
            self.choose_input_button.config(state=tk.NORMAL)
            self.create_input_button.config(state=tk.DISABLED)
        else:
            self.choose_input_button.config(state=tk.DISABLED)
            self.create_input_button.config(state=tk.NORMAL)

    def choose_labels_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            if os.path.exists('train_labels.json'):
                if messagebox.askyesno("File Exists", "train_labels.json already exists. Do you want to replace it?"):
                    self.copy_file(file_path, 'train_labels.json')
            else:
                self.copy_file(file_path, 'train_labels.json')

    def copy_file(self, source, destination):
        shutil.copy2(source, destination)
        messagebox.showinfo("Success", f"File copied to {destination}")

    def generate_labels(self):
        LabelGenerator(self.master)

    def start_training(self):
        # Backup existing trained_model_1 directory if it exists
        if os.path.exists("trained_model_1"):
            backup_name = "trained_model_1_backup"
            counter = 1
            while os.path.exists(f"{backup_name}_{counter}"):
                counter += 1
            shutil.move("trained_model_1", f"{backup_name}_{counter}")
            messagebox.showinfo("Backup Created", f"Existing model backed up as {backup_name}_{counter}")

        progress_window = tk.Toplevel(self.master)
        progress_window.title("Training Progress")
        progress_label = tk.Label(progress_window, text="Initializing training...")
        progress_label.pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, length=300, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start()

        def run_training():
            try:
                process = subprocess.Popen([os.path.join('venv', 'bin', 'python'), "distilbert_train.py"],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                
                start_time = time.time()
                real_progress_started = False

                def update_progress():
                    nonlocal real_progress_started
                    if process.poll() is None:  # If the process is still running
                        output = process.stdout.readline().strip()
                        if output:
                            if "%" in output and "|" in output and "Map" not in output:
                                if not real_progress_started:
                                    real_progress_started = True
                                    progress_bar.stop()
                                    progress_bar.config(mode='determinate')
                                percentage = int(output.split("|")[0].strip().replace("%", ""))
                                progress_bar['value'] = percentage
                                elapsed_time = int(time.time() - start_time)
                                estimated_total_time = int(elapsed_time * 100 / percentage) if percentage > 0 else 0
                                remaining_time = max(0, estimated_total_time - elapsed_time)
                                progress_label.config(text=f"Training progress: {percentage}% (ETA: {remaining_time//60}m {remaining_time%60}s)")
                            elif not real_progress_started:
                                progress_label.config(text="Training in progress...")
                        progress_window.after(100, update_progress)
                    else:
                        progress_window.destroy()
                        if process.returncode == 0:
                            messagebox.showinfo("Training Complete", "Model training has finished successfully.")
                        else:
                            messagebox.showerror("Training Error", "An error occurred during training. Check the console for details.")

                update_progress()

            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Error", f"An error occurred: {str(e)}")

        # Start the training process in a separate thread
        threading.Thread(target=run_training, daemon=True).start()

    def choose_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Markdown files", "*.md")])
        if file_path:
            if os.path.exists('infer_input.md'):
                if messagebox.askyesno("File Exists", "infer_input.md already exists. Do you want to replace it?"):
                    self.copy_file(file_path, 'infer_input.md')
            else:
                self.copy_file(file_path, 'infer_input.md')

    def create_input_file(self):
        input_window = tk.Toplevel(self.master)
        input_window.title("Create or Edit Input File")
        input_window.geometry("600x400")

        text_area = tk.Text(input_window, wrap=tk.WORD)
        text_area.pack(expand=True, fill='both', padx=10, pady=10)

        # Load existing content if file exists
        if os.path.exists('infer_input.md'):
            with open('infer_input.md', 'r') as f:
                existing_content = f.read()
                text_area.insert(tk.END, existing_content)

        def save_input():
            content = text_area.get("1.0", tk.END)
            with open('infer_input.md', 'w') as f:
                f.write(content)
            messagebox.showinfo("Success", "Input file saved as infer_input.md")
            input_window.destroy()

        save_button = tk.Button(input_window, text="Save", command=save_input)
        save_button.pack(pady=10)

    def start_inferring(self):
        progress_window = tk.Toplevel(self.master)
        progress_window.title("Inferring Progress")
        progress_label = tk.Label(progress_window, text="Processing paragraphs...")
        progress_label.pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
        progress_bar.pack(pady=10)

        def run_inferring():
            command = [os.path.join('venv', 'bin', 'python'), "distilbert_infer.py", "-s"]
            if self.top_label_only.get():
                command.append("-l")
            
            process = subprocess.Popen(command, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.STDOUT,
                                       universal_newlines=True)
            
            for line in process.stdout:
                try:
                    if line.startswith("Processing paragraphs:"):
                        progress = int(line.split("|")[0].split(":")[1].strip().rstrip('%'))
                        progress_bar['value'] = progress
                        progress_window.update()
                except:
                    # If parsing fails, show an indeterminate progress bar
                    progress_bar.config(mode='indeterminate')
                    progress_bar.start()
                    progress_window.update()

            process.wait()
            progress_window.destroy()
            messagebox.showinfo("Inferring Complete", "Model inferring has finished.")

        self.master.after(100, run_inferring)

    def start_validation(self):
        progress_window = tk.Toplevel(self.master)
        progress_window.title("Validation Progress")
        progress_label = tk.Label(progress_window, text="Validating...")
        progress_label.pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, length=300, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start()

        def run_validation():
            process = subprocess.Popen([os.path.join('venv', 'bin', 'python'), "distilbert_validate.py"], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.STDOUT,
                                       universal_newlines=True)
            
            output = process.communicate()[0]
            progress_window.destroy()
            
            self.validation_output.delete('1.0', tk.END)
            self.validation_output.insert(tk.END, output)
            
            messagebox.showinfo("Validation Complete", "Validation has finished. Results are displayed in the Validation frame.")

        self.master.after(100, run_validation)

    def assign_buckets(self):
        if os.path.exists('labeling.html'):
            webbrowser.open('file://' + os.path.realpath('labeling.html'))
        else:
            messagebox.showerror("Error", "labeling.html not found in the current directory.")

    def show_help(self):
        help_window = tk.Toplevel(self.master)
        help_window.title("Help")
        help_window.geometry("600x400")

        help_text = tk.Text(help_window, wrap=tk.WORD)
        help_text.pack(expand=True, fill='both', padx=10, pady=10)

        if os.path.exists('README.md'):
            with open('README.md', 'r') as f:
                readme_content = f.read()
            help_text.insert(tk.END, readme_content)
        else:
            help_text.insert(tk.END, "README.md not found in the current directory.")

        help_text.config(state=tk.DISABLED)

class LabelGenerator:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Generate Labels")
        self.labels = {}
        self.create_widgets()
        self.load_existing_labels()
        self.selected_label = None

    def create_widgets(self):
        tk.Label(self.window, text="Label:").grid(row=0, column=0, padx=5, pady=5)
        self.label_entry = tk.Entry(self.window)
        self.label_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(self.window, text="Choose Color", command=self.choose_color).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(self.window, text="Increase if:").grid(row=1, column=0, padx=5, pady=5)
        self.increase_entry = tk.Entry(self.window)
        self.increase_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        tk.Label(self.window, text="Decrease if:").grid(row=2, column=0, padx=5, pady=5)
        self.decrease_entry = tk.Entry(self.window)
        self.decrease_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        tk.Label(self.window, text="Must have:").grid(row=3, column=0, padx=5, pady=5)
        self.must_have_entry = tk.Entry(self.window)
        self.must_have_entry.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        self.add_edit_button = tk.Button(self.window, text="Add Label", command=self.add_or_edit_label)
        self.add_edit_button.grid(row=4, column=0, pady=10)
        
        self.add_new_button = tk.Button(self.window, text="Add as New", command=self.add_as_new, state=tk.DISABLED)
        self.add_new_button.grid(row=4, column=1, pady=10)
        
        tk.Button(self.window, text="Remove Label", command=self.remove_label).grid(row=4, column=2, pady=10)

        self.labels_listbox = tk.Listbox(self.window, width=50)
        self.labels_listbox.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        self.labels_listbox.bind('<<ListboxSelect>>', self.on_select)

        tk.Button(self.window, text="Save Labels", command=self.save_labels).grid(row=6, column=0, columnspan=3, pady=10)

    def load_existing_labels(self):
        if os.path.exists('train_labels.json'):
            with open('train_labels.json', 'r') as f:
                self.labels = json.load(f)
            for label_id, label_data in self.labels.items():
                self.labels_listbox.insert(tk.END, f"{label_data['label']} - {label_data['color']}")

    def choose_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.color = color

    def on_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            self.selected_label = str(index)
            label_data = self.labels[self.selected_label]
            self.label_entry.delete(0, tk.END)
            self.label_entry.insert(0, label_data['label'])
            self.color = label_data['color']
            self.increase_entry.delete(0, tk.END)
            self.increase_entry.insert(0, ','.join(label_data['increase_if']))
            self.decrease_entry.delete(0, tk.END)
            self.decrease_entry.insert(0, ','.join(label_data['decrease_if']))
            self.must_have_entry.delete(0, tk.END)
            self.must_have_entry.insert(0, ','.join(label_data['must_have']))
            self.add_edit_button.config(text="Edit Label")
            self.add_new_button.config(state=tk.NORMAL)
        else:
            self.selected_label = None
            self.add_edit_button.config(text="Add Label")
            self.add_new_button.config(state=tk.DISABLED)

    def add_or_edit_label(self):
        label = self.label_entry.get()
        if label and hasattr(self, 'color'):
            if self.selected_label is not None:
                # Editing existing label
                self.labels[self.selected_label] = {
                    "label": label,
                    "color": self.color,
                    "increase_if": self.increase_entry.get().split(',') if self.increase_entry.get() else [],
                    "decrease_if": self.decrease_entry.get().split(',') if self.decrease_entry.get() else [],
                    "must_have": self.must_have_entry.get().split(',') if self.must_have_entry.get() else []
                }
                self.labels_listbox.delete(int(self.selected_label))
                self.labels_listbox.insert(int(self.selected_label), f"{label} - {self.color}")
            else:
                # Adding new label
                if not any(label_data['label'] == label for label_data in self.labels.values()):
                    new_id = str(len(self.labels))
                    self.labels[new_id] = {
                        "label": label,
                        "color": self.color,
                        "increase_if": self.increase_entry.get().split(',') if self.increase_entry.get() else [],
                        "decrease_if": self.decrease_entry.get().split(',') if self.decrease_entry.get() else [],
                        "must_have": self.must_have_entry.get().split(',') if self.must_have_entry.get() else []
                    }
                    self.labels_listbox.insert(tk.END, f"{label} - {self.color}")
                else:
                    messagebox.showerror("Error", "Label already exists. Choose a unique label name.")
                    return
            self.clear_entries()

    def add_as_new(self):
        label = self.label_entry.get()
        if label and hasattr(self, 'color'):
            if not any(label_data['label'] == label for label_data in self.labels.values()):
                new_id = str(len(self.labels))
                self.labels[new_id] = {
                    "label": label,
                    "color": self.color,
                    "increase_if": self.increase_entry.get().split(',') if self.increase_entry.get() else [],
                    "decrease_if": self.decrease_entry.get().split(',') if self.decrease_entry.get() else [],
                    "must_have": self.must_have_entry.get().split(',') if self.must_have_entry.get() else []
                }
                self.labels_listbox.insert(tk.END, f"{label} - {self.color}")
                self.clear_entries()
            else:
                messagebox.showerror("Error", "Label already exists. Choose a unique label name.")

    def remove_label(self):
        selected = self.labels_listbox.curselection()
        if selected:
            index = selected[0]
            self.labels_listbox.delete(index)
            del self.labels[str(index)]
            # Renumber the remaining labels
            self.labels = {str(i): label for i, label in enumerate(self.labels.values())}
            self.clear_entries()
            self.selected_label = None
            self.add_edit_button.config(text="Add Label")
            self.add_new_button.config(state=tk.DISABLED)

    def clear_entries(self):
        self.label_entry.delete(0, tk.END)
        self.increase_entry.delete(0, tk.END)
        self.decrease_entry.delete(0, tk.END)
        self.must_have_entry.delete(0, tk.END)
        if hasattr(self, 'color'):
            del self.color

    def save_labels(self):
        with open('train_labels.json', 'w') as f:
            json.dump(self.labels, f)
        messagebox.showinfo("Success", "Labels saved to train_labels.json")
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = NoteSort(root)
    root.mainloop()

