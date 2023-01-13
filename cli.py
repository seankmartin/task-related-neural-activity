from your_package.your_module import TemplateClass

def main():
    first = TemplateClass(5, 2)
    second = TemplateClass(10, 4)

    print("Wow I got a result of {} adding these".format(first + second))

if __name__ == "__main__":
    main()