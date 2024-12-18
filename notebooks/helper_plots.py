import matplotlib.pyplot as plt


def violin_plot(
    list_price, list_number_simulations, benchmark, title, price_training=None
):
    """perform violin plot of a pricer for various timesteps

    Parameters
    ----------
    list_price : _type_
        matrix of prices
    list_number_simulations : _type_
        list of Monte-Carlo scenarios numbers
    benchmark : _type_
        price benchmark
    title : _type_
        title of the plot
    price_training : _type_, optional
        _description_, by default None
    """
    plt.figure(figsize=(9, 8))

    valeur_theo = "valeur benchmark = {:.2f}".format(benchmark)  # contenu de legend
    plt.axhline(y=benchmark, color="red", label=str(valeur_theo))  # ligne horizontale

    if price_training is not None:
        training = "price on training data = {:.2f}".format(
            price_training
        )  # contenu de legend
        plt.axhline(
            y=price_training, color="green", label=str(training)
        )  # ligne horizontale

    data = [list_price[:, idx] for idx in range(list_price.shape[1])]

    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks([y + 1 for y in range(len(data))], labels=list_number_simulations)

    plt.xlabel("N", fontsize=15)
    plt.ylabel("${P}_{N}$", fontsize=15)
    plt.title(title)

    plt.legend()
    plt.show()
