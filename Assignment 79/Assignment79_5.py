
import tkinter as tk

def login():
    username = username_entry.get()
    password = password_entry.get()

    if username == "admin" and password == "12345":
        result_label.config(text="Login Successful!")
    else:
        result_label.config(text="Invalid username or password!")

root = tk.Tk()
root.title("Basic Login Form")
root.geometry("350x250")

tk.Label(root, text="Username:").pack(pady=5)
username_entry = tk.Entry(root)
username_entry.pack()

tk.Label(root, text="Password:").pack(pady=5)
password_entry = tk.Entry(root, show="*")
password_entry.pack()

tk.Button(root, text="Login", command=login).pack(pady=15)

result_label = tk.Label(root, text="Please login")
result_label.pack()

root.mainloop()