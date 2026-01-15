def validate(form):
    for key, value in form.items():
        try:
            float(value)
        except ValueError:
            return f"Invlid value for {key}"
    return None