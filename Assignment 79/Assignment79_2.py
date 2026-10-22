
import tkinter as tk

def find_largest():
    try:
        a = float(entry1.get())
        b = float(entry2.get())
        c = float(entry3.get())

        result_label.config(
            text=f"Largest Number = {max(a, b, c)}"
        )
    except ValueError:
        result_label.config(text="Please enter valid numbers!")

root = tk.Tk()
root.title("Find the Largest Number")
root.geometry("350x300")

tk.Label(root, text="Enter First Number:").pack(pady=5)
entry1 = tk.Entry(root)
entry1.pack()

tk.Label(root, text="Enter Second Number:").pack(pady=5)
entry2 = tk.Entry(root)
entry2.pack()

tk.Label(root, text="Enter Third Number:").pack(pady=5)
entry3 = tk.Entry(root)
entry3.pack()

tk.Button(
    root, text="Find Maximum", command=find_largest
).pack(pady=15)

result_label = tk.Label(root, text="Result will appear here")
result_label.pack()

root.mainloop()