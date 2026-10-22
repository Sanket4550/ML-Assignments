
import tkinter as tk

def check_eligibility():
    name = name_entry.get().strip()

    try:
        age = int(age_entry.get())

        if not name:
            result_label.config(text="Please enter your name!")
        elif age < 0:
            result_label.config(text="Age cannot be negative!")
        elif age >= 18:
            result_label.config(
                text=f"{name}, you are eligible to vote."
            )
        else:
            result_label.config(
                text=f"{name}, you are not eligible to vote."
            )
    except ValueError:
        result_label.config(text="Please enter a valid age!")

root = tk.Tk()
root.title("Voting Eligibility Checker")
root.geometry("400x250")

tk.Label(root, text="Enter Your Name:").pack(pady=5)
name_entry = tk.Entry(root)
name_entry.pack()

tk.Label(root, text="Enter Your Age:").pack(pady=5)
age_entry = tk.Entry(root)
age_entry.pack()

tk.Button(
    root, text="Check Eligibility", command=check_eligibility
).pack(pady=15)

result_label = tk.Label(root, text="Result will appear here")
result_label.pack()

root.mainloop()