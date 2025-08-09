import numpy as np

from code.semester_project.common.types import Price


class PriceAttribution:
    def __init__(
        self,
        product_predictions,
        price_tag_predictions,
        product_info=None,
        remove_empty_prices=False,
    ):
        """Takes in product and price tag predictions, and matches prices to products

        :param product_predictions: [
            {
                label: str,
                min_x: int,
                min_y: int,
                max_x: int,
                max_y: int,
            },
            ...
        ]
        :param price_tag_predictions: [
            {
                price_tag_text: str,
                label: str,
                min_x: int,
                min_y: int,
                max_x: int,
                max_y: int,
            },
            ...
        ]
        :param product_info: {
            upc: {
                name: str,
                brand: str,
                packaging: {
                    size: float,
                    quantity: int,
                    container: str,
                    unit: str,
                }
            },
            ...
        }
        """
        self.product_info = product_info or {}
        self.product_predictions = product_predictions
        self.price_tag_predictions = self._clean_price_tag_predictions(
            price_tag_predictions
        )

        self.similar_products = self.get_similar_products()
        self.product_matrix, self.product_labels = self.get_matrix(
            self.product_predictions
        )
        self.price_tag_matrix, self.price_tag_labels = self.get_matrix(
            self.price_tag_predictions
        )
        self.price_tag_texts = [
            pred["price_tag_text"] for pred in self.price_tag_predictions
        ]
        self.upcs = [pred["label"] for pred in self.product_predictions]
        if remove_empty_prices:
            self.filter_to_valid_prices()

    def _clean_price_tag_predictions(
        self, price_tag_predictions: list[dict[str, str | float]]
    ):
        """Removes nonsense price tag labels (e.g. 'null')"""
        nonsense_labels = ["null"]
        return [
            pred
            for pred in price_tag_predictions
            if pred["price_tag_text"] not in nonsense_labels
        ]

    def get_similar_products(self):
        similar_products = []
        for upc, product in self.product_info.items():
            if product.get("done", False):
                continue
            similar_products.append([upc])
            others = {
                other_upc: other_product
                for other_upc, other_product in self.product_info.items()
                if (other_upc != upc and not other_product.get("done", False))
            }
            for other_upc, other_product in others.items():
                packaging = product["packaging"]
                other_packaging = other_product["packaging"]
                if (
                    other_product["brand"] == product["brand"]
                    and other_packaging == packaging
                ):
                    similar_products[-1].append(other_upc)
                    other_product["done"] = True
        return similar_products

    def get_matrix(self, predictions) -> tuple[np.ndarray, list[str]]:
        matrix = np.array(
            [
                [
                    pred["min_x"],
                    pred["min_y"],
                    pred["max_x"],
                    pred["max_y"],
                ]
                for pred in predictions
            ]
        )
        labels = [pred["label"] for pred in predictions]
        return matrix, labels

    def filter_to_indices(self, array, indices):
        return [value for i, value in enumerate(array) if i in indices]

    def filter_to_valid_prices(self):
        parsed_prices = self.price_tag_texts
        valid_inds = [
            i
            for i in range(len(parsed_prices))
            if (parsed_prices[i] and len(parsed_prices[i]) > 0)
        ]
        self.price_tag_predictions = self.filter_to_indices(
            self.price_tag_predictions, valid_inds
        )
        self.price_tag_labels = self.filter_to_indices(
            self.price_tag_labels, valid_inds
        )
        self.price_tag_texts = self.filter_to_indices(self.price_tag_texts, valid_inds)
        self.price_tag_matrix = (
            self.price_tag_matrix[valid_inds, :] if valid_inds else []
        )

    def get_valid_prices(self):
        return [price for price in self.price_tag_texts if price]

    @staticmethod
    def get_nearest_ind(centroid, others):
        x_ind = 0
        y_ind = 1
        x = np.abs(others[:, x_ind] - centroid[x_ind]) ** 2
        y = np.abs(others[:, y_ind] - centroid[y_ind]) ** 2
        distances = np.sqrt(x + y)
        return distances.argmin()

    def run(self):
        if len(self.price_tag_matrix) == 0:
            return {pred["label"]: None for pred in self.product_predictions}
        product_prices = {}
        price_tag_centroids = (
            self.price_tag_matrix[:, 2:] + self.price_tag_matrix[:, :2]
        ) / 2
        for i, product in enumerate(self.product_matrix):
            centroid = (product[2:] + product[:2]) / 2
            nearest_ind = self.get_nearest_ind(centroid, price_tag_centroids)
            price = self.price_tag_texts[nearest_ind]
            if not price:
                price = None
            upc = self.product_predictions[i]["label"]
            product_prices[upc] = price
        return product_prices


class NearestPriceAttribution(PriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions):
        super(NearestPriceAttribution, self).__init__(
            product_predictions, price_tag_predictions, product_info={}
        )

    def run(self) -> dict[str, str | None]:
        if len(self.price_tag_matrix) == 0:
            return {pred["label"]: None for pred in self.product_predictions}
        product_prices = {}
        price_tag_centroids = (
            self.price_tag_matrix[:, 2:] + self.price_tag_matrix[:, :2]
        ) / 2
        for i, product in enumerate(self.product_matrix):
            centroid = (product[2:] + product[:2]) / 2
            centroid[1] += (
                product[3] - product[1]
            ) * 0.25  # Use point lower on box to avoid matching the wrong price tags
            upc = self.product_predictions[i]["label"]
            nearest_ind = self.get_nearest_ind(centroid, price_tag_centroids)
            price = (
                self.price_tag_texts[nearest_ind] if nearest_ind is not None else None
            ) or None
            product_prices[upc] = price
        return product_prices


class NearestToProductGroupAttribution(NearestPriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions):
        super(NearestToProductGroupAttribution, self).__init__(
            product_predictions, price_tag_predictions
        )

    def run(self):
        if len(self.price_tag_matrix) == 0:
            return {pred["label"]: None for pred in self.product_predictions}
        product_prices = {}
        price_tag_centroids = (
            self.price_tag_matrix[:, 2:] + self.price_tag_matrix[:, :2]
        ) / 2
        for upc in set(self.product_labels):
            product_boxes = self.product_matrix[np.array(self.product_labels) == upc]
            product_centroids = (product_boxes[:, 2:] + product_boxes[:, :2]) / 2
            group_centroid = product_centroids.mean(axis=0)
            nearest_ind = self.get_nearest_ind(group_centroid, price_tag_centroids)
            price = (
                self.price_tag_texts[nearest_ind] if nearest_ind is not None else None
            ) or None
            product_prices[upc] = price
        return product_prices


class NearestByProductVoteAttribution(PriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions):
        super(NearestByProductVoteAttribution, self).__init__(
            product_predictions, price_tag_predictions, product_info={}
        )

    @staticmethod
    def get_nearest_ind(product_centroids, others):
        x_ind = 0
        y_ind = 1
        voted_inds = []
        for centroid in product_centroids:
            x = np.abs(others[:, x_ind] - centroid[x_ind]) ** 2
            y = np.abs(others[:, y_ind] - centroid[y_ind]) ** 2
            distances = np.sqrt(x + y)
            voted_inds.append(distances.argmin())
        voted_ind = max(set(voted_inds), key=voted_inds.count)
        return voted_ind

    def run(self):
        if len(self.price_tag_matrix) == 0:
            return {pred["label"]: None for pred in self.product_predictions}
        product_prices = {}
        price_tag_centroids = (
            self.price_tag_matrix[:, 2:] + self.price_tag_matrix[:, :2]
        ) / 2
        for upc in set(self.product_labels):
            product_boxes = self.product_matrix[np.array(self.product_labels) == upc]
            product_centroids = (product_boxes[:, 2:] + product_boxes[:, :2]) / 2
            nearest_ind = self.get_nearest_ind(product_centroids, price_tag_centroids)
            price = (
                self.price_tag_texts[nearest_ind] if nearest_ind is not None else None
            ) or None
            product_prices[upc] = price
        return product_prices


class NearestByProductGroupCentroidTopKVoting(PriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions):
        super(NearestByProductGroupCentroidTopKVoting, self).__init__(
            product_predictions, price_tag_predictions, product_info={}
        )

    @staticmethod
    def get_nearest_ind(centroid, others):
        x_ind = 0
        y_ind = 1
        x = np.abs(others[:, x_ind] - centroid[x_ind]) ** 2
        y = np.abs(others[:, y_ind] - centroid[y_ind]) ** 2
        distances = np.sqrt(x + y)
        closest_inds = distances.argsort()[:5]
        return max(set(closest_inds), key=list(closest_inds).count)

    def run(self):
        if len(self.price_tag_matrix) == 0:
            return {pred["label"]: None for pred in self.product_predictions}
        product_prices = {}
        price_tag_centroids = (
            self.price_tag_matrix[:, 2:] + self.price_tag_matrix[:, :2]
        ) / 2
        for upc in set(self.product_labels):
            product_boxes = self.product_matrix[np.array(self.product_labels) == upc]
            product_centroids = (product_boxes[:, 2:] + product_boxes[:, :2]) / 2
            product_group_centroid = product_centroids.mean(axis=0)
            nearest_ind = self.get_nearest_ind(
                product_group_centroid, price_tag_centroids
            )
            price = (
                self.price_tag_texts[nearest_ind] if nearest_ind is not None else None
            ) or None
            product_prices[upc] = price
        return product_prices


class NearestBelowPriceAttribution(NearestPriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions):
        super(NearestBelowPriceAttribution, self).__init__(
            product_predictions, price_tag_predictions
        )

    @staticmethod
    def get_nearest_ind(centroid, others) -> int | None:
        x_ind = 0
        y_ind = 1
        filtered_others = others[others[:, y_ind] > centroid[y_ind]]
        if len(filtered_others) == 0:
            return None
        x = np.abs(filtered_others[:, x_ind] - centroid[x_ind]) ** 2
        y = np.abs(filtered_others[:, y_ind] - centroid[y_ind]) ** 2
        distances = np.sqrt(x + y)
        filtered_ind = distances.argmin()
        voted_centroid = filtered_others[filtered_ind]
        voted_ind = np.where((others == voted_centroid).all(axis=1))[0][0]
        return voted_ind


class NearestBelowWithinPkgGroupAttribution(NearestBelowPriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions, upc_to_group_upcs):
        super(NearestBelowPriceAttribution, self).__init__(
            product_predictions, price_tag_predictions
        )
        self.upc_to_group_upcs = upc_to_group_upcs

    def run(self) -> dict[str, Price]:
        if len(self.price_tag_matrix) == 0:
            return {pred["label"]: None for pred in self.product_predictions}
        product_prices = {}
        price_tag_centroids = (
            self.price_tag_matrix[:, 2:] + self.price_tag_matrix[:, :2]
        ) / 2
        for i, product in enumerate(self.product_matrix):
            centroid = (product[2:] + product[:2]) / 2
            centroid[1] += (
                product[3] - product[1]
            ) * 0.25  # Use point lower on box to avoid matching the wrong price tags
            upc = self.product_predictions[i]["label"]

            group_product_boxes, other_product_boxes = self.get_group_and_other_indices(
                upc
            )

            if len(group_product_boxes) == 0:
                product_prices[upc] = None
                continue
            min_x = group_product_boxes[:, 0].min()
            max_x = group_product_boxes[:, 2].max()
            min_y = group_product_boxes[:, 1].min()
            max_y = group_product_boxes[:, 3].max()
            height = max_y - min_y

            centroid = self.adjust_centroid(
                centroid, other_product_boxes, height, min_x, max_x, min_y
            )

            # Set products to the left and right of the group to (-1,-1) to exclude them
            tag_centroids_copy = price_tag_centroids.copy()
            tag_centroids_copy[tag_centroids_copy[:, 0] < min_x] = -1
            tag_centroids_copy[tag_centroids_copy[:, 0] > max_x] = -1

            # Get products below, to exclude price tags in or below groups below this one
            products_below = other_product_boxes[other_product_boxes[:, 1] > max_y]
            products_below_within_min_x = products_below[products_below[:, 2] > min_x]
            products_below_within = products_below_within_min_x[
                products_below_within_min_x[:, 1] < max_x
            ]
            if len(products_below_within) > 0:
                top_box_index = products_below_within[:, 1].argmin()
                top_box = products_below_within[top_box_index]
                top_centroid = (top_box[2:] + top_box[:2]) / 2
                tag_centroids_copy[tag_centroids_copy[:, 1] > top_centroid[1]] = -1

            # Prioritize price tags that are inside the product group bounds
            tag_centroid_inds_within = np.nonzero(
                np.logical_and(
                    tag_centroids_copy[:, 1] > min_y, tag_centroids_copy[:, 1] < max_y
                )
            )[0]
            if len(tag_centroid_inds_within) == 1:
                nearest_ind = tag_centroid_inds_within[0]
            else:
                nearest_inds = self.get_nearest_inds(centroid, tag_centroids_copy)
                nearest_ind = nearest_inds[0]
                tag_text = (
                    self.price_tag_texts[nearest_ind]
                    if nearest_ind is not None
                    else None
                )
                if nearest_inds[0] is None or tag_text is None:
                    nearest_ind = nearest_inds[1]
            price = (
                self.price_tag_texts[nearest_ind] if nearest_ind is not None else None
            ) or None
            product_prices[upc] = price
        return product_prices

    def get_group_and_other_indices(self, upc):
        # Get group boxes
        group_upcs = self.upc_to_group_upcs.get(upc, [])
        group_indices = [
            i for i, label in enumerate(self.product_labels) if label in group_upcs
        ]
        group_product_boxes = self.product_matrix[group_indices]

        # Get out-of-group boxes
        other_indices = [
            i for i, _ in enumerate(self.product_labels) if i not in group_indices
        ]
        other_product_boxes = self.product_matrix[other_indices]
        return group_product_boxes, other_product_boxes

    def adjust_centroid(
        self, centroid, other_product_boxes, height, min_x, max_x, min_y
    ):
        # Get products directly above, within min_x/max_x
        products_above = other_product_boxes[other_product_boxes[:, 1] < min_y]
        products_above_within_min_x = products_above[products_above[:, 0] > min_x]
        products_above_within = products_above_within_min_x[
            products_above_within_min_x[:, 2] < max_x
        ]
        if len(products_above_within) > 0:
            # If no products above, move centroid up to include higher price tags
            centroid[1] -= height * 0.25
        return centroid

    @staticmethod
    def get_nearest_inds(centroid, tag_centroids) -> list[int | None]:
        x_ind = 0
        y_ind = 1
        filtered_others = tag_centroids[tag_centroids[:, y_ind] > centroid[y_ind]]
        if len(filtered_others) == 0:
            return [None, None]
        x = np.abs(filtered_others[:, x_ind] - centroid[x_ind]) ** 2
        y = np.abs(filtered_others[:, y_ind] - centroid[y_ind]) ** 2
        distances = np.sqrt(x + y)
        filtered_inds = distances.argsort()[:2]
        voted_inds = []
        for ind in filtered_inds:
            voted_centroid = filtered_others[ind]
            voted_ind = np.where((tag_centroids == voted_centroid).all(axis=1))[0][0]
            voted_inds.append(voted_ind)
        while len(voted_inds) < 2:
            voted_inds.append(None)
        return voted_inds


class NearestBelowWithVotingPriceAttribution(NearestByProductVoteAttribution):
    def __init__(self, product_predictions, price_tag_predictions):
        super(NearestBelowWithVotingPriceAttribution, self).__init__(
            product_predictions, price_tag_predictions
        )

    @staticmethod
    def get_nearest_ind(centroids, others):
        x_ind = 0
        y_ind = 1
        voted_inds = []
        for voted_centroid in centroids:
            filtered_others = others[others[:, y_ind] > voted_centroid[y_ind]]
            if len(filtered_others) == 0:
                continue
            x = np.abs(filtered_others[:, x_ind] - voted_centroid[x_ind]) ** 2
            y = np.abs(filtered_others[:, y_ind] - voted_centroid[y_ind]) ** 2
            distances = np.sqrt(x + y)
            filtered_ind = distances.argmin()
            voted_centroid = filtered_others[filtered_ind]
            voted_ind = np.where((others == voted_centroid).all(axis=1))[0][0]
            voted_inds.append(voted_ind)
        final_voted_ind = (
            max(set(voted_inds), key=voted_inds.count) if voted_inds else None
        )
        return final_voted_ind


class NearestWithPriceFilterAttribution(PriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions, product_info=None):
        product_info = product_info or {}
        super(NearestWithPriceFilterAttribution, self).__init__(
            product_predictions, price_tag_predictions, product_info
        )

    @staticmethod
    def get_nearest_ind(centroid, others):
        x_ind = 0
        y_ind = 1
        x = np.abs(others[:, x_ind] - centroid[x_ind]) ** 2
        y = np.abs(others[:, y_ind] - centroid[y_ind]) ** 2
        distances = np.sqrt(x + y)
        return distances.argmin() if len(distances) > 0 else None

    def filter_price_tags(self):
        min_x_dim = 0
        max_x_dim = 2
        product_min_x = self.product_matrix[:, min_x_dim].min()
        product_max_x = self.product_matrix[:, max_x_dim].max()
        # Filter out price tags that are completely to the left or right of the products
        self.price_tag_matrix = self.price_tag_matrix[
            self.price_tag_matrix[:, max_x_dim] >= product_min_x
        ]
        self.price_tag_matrix = self.price_tag_matrix[
            self.price_tag_matrix[:, min_x_dim] <= product_max_x
        ]

    def run(self):
        product_prices = {pred["label"]: None for pred in self.product_predictions}
        if len(self.price_tag_matrix) == 0:
            return product_prices
        self.filter_price_tags()
        price_tag_centroids = (
            self.price_tag_matrix[:, 2:] + self.price_tag_matrix[:, :2]
        ) / 2
        for i, product in enumerate(self.product_matrix):
            centroid = (product[2:] + product[:2]) / 2
            nearest_ind = self.get_nearest_ind(centroid, price_tag_centroids)
            if nearest_ind is not None:
                price = self.price_tag_texts[nearest_ind]
                if price:
                    upc = self.product_predictions[i]["label"]
                    product_prices[upc] = price
        return product_prices


class TopDownNearestAttribution(PriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions, product_info=None):
        product_info = product_info or {}
        super(TopDownNearestAttribution, self).__init__(
            product_predictions, price_tag_predictions, product_info
        )

    @staticmethod
    def get_nearest_ind(centroid, others):
        x_ind = 0
        y_ind = 1
        x = np.abs(others[:, x_ind] - centroid[x_ind]) ** 2
        y = np.abs(others[:, y_ind] - centroid[y_ind]) ** 2
        distances = np.sqrt(x + y)
        return distances.argmin() if len(distances) > 0 else None

    def filter_price_tags(self):
        min_x_dim = 0
        max_x_dim = 2
        product_min_x = self.product_matrix[:, min_x_dim].min()
        product_max_x = self.product_matrix[:, max_x_dim].max()
        # Filter out price tags that are completely to the left or right of the products
        self.price_tag_matrix = self.price_tag_matrix[
            self.price_tag_matrix[:, max_x_dim] >= product_min_x
        ]
        self.price_tag_matrix = self.price_tag_matrix[
            self.price_tag_matrix[:, min_x_dim] <= product_max_x
        ]

    def run(self):
        product_prices = {pred["label"]: None for pred in self.product_predictions}
        if len(self.price_tag_matrix) == 0:
            return product_prices
        # for upc_set in self.similar_products:
        #    for upc in upc_set:
        #        product_index = self.upcs.index(upc)
        #        pass
        # For each product, match it to the nearest price tag
        self.filter_price_tags()
        price_tag_centroids = (
            self.price_tag_matrix[:, 2:] + self.price_tag_matrix[:, :2]
        ) / 2
        for i, product in enumerate(self.product_matrix):
            centroid = (product[2:] + product[:2]) / 2
            nearest_ind = self.get_nearest_ind(centroid, price_tag_centroids)
            if nearest_ind is not None:
                price = self.price_tag_texts[nearest_ind]
                if price:
                    upc = self.product_predictions[i]["label"]
                    product_prices[upc] = price
        return product_prices


class PerfectPriceAttribution(PriceAttribution):
    def __init__(
        self,
        product_predictions,
        price_tag_predictions,
        product_price_labels,
        product_info=None,
    ):
        """Price Attribution accuracy upper bound by correctly assigning prices if the price was extracted correctly.
        This shows how much of the potential accuracy gain is due to the extraction piece
        """
        self.product_price_labels = product_price_labels
        product_info = product_info or {}
        super(PerfectPriceAttribution, self).__init__(
            product_predictions, price_tag_predictions, product_info
        )

    def run(self):
        product_prices = {}
        for upc, prices in self.product_price_labels.items():
            for text in self.price_tag_texts:
                if text and text in prices:
                    product_prices[upc] = text
        for upc in self.product_price_labels:
            if upc not in product_prices:
                product_prices[upc] = None
        return product_prices


class ValidPriceAttribution(PriceAttribution):
    def __init__(self, product_predictions, price_tag_predictions, product_info=None):
        super(ValidPriceAttribution, self).__init__(
            product_predictions, price_tag_predictions, product_info
        )

    def run(self):
        pass
