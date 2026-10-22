
import tkinter as tk

def add_numbers():
    try:
        a = float(entry1.get())
        b = float(entry2.get())
        result_label.config(text=f"Sum = {a + b}")
    except ValueError:
        result_label.config(text="Please enter valid numbers!")

root = tk.Tk()
root.title("Addition of Two Numbers")
root.geometry("350x250")

tk.Label(root, text="Enter First Number:").pack(pady=5)
entry1 = tk.Entry(root)
entry1.pack()

tk.Label(root, text="Enter Second Number:").pack(pady=5)
entry2 = tk.Entry(root)
entry2.pack()

tk.Button(root, text="Add", command=add_numbers).pack(pady=15)

result_label = tk.Label(root, text="Result will appear here")
result_label.pack()

root.mainloop()