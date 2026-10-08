import unittest

from shapely.geometry import Polygon

from evacusim.visualization.train_geometry import compute_train_polygons


class TrainGeometryTests(unittest.TestCase):
    def test_builds_one_offset_rectangle_per_platform(self):
        geometry = {
            "levels": {
                "level_-1": {
                    "walkable_areas": {
                        "floor": [(-10, -10), (20, -10), (20, 20), (-10, 20)],
                        "platform_1": [(0, 0), (2, 0), (2, 10), (0, 10)],
                        "platform_2": [(5, 0), (15, 0), (15, 2), (5, 2)],
                    }
                }
            }
        }

        trains = compute_train_polygons(geometry)

        self.assertEqual(set(trains), {"train_platform_1", "train_platform_2"})
        self.assertTrue(all(len(coords) == 4 for coords in trains.values()))

    def test_train_prefers_side_outside_walkable_floor(self):
        floor = Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])
        geometry = {
            "levels": {
                "level_-1": {
                    "walkable_areas": {
                        "floor": list(floor.exterior.coords),
                        "platform_1": [(0, 2), (2, 2), (2, 18), (0, 18)],
                    }
                }
            }
        }

        train = Polygon(compute_train_polygons(geometry)["train_platform_1"])

        self.assertEqual(train.intersection(floor).area, 0.0)


if __name__ == "__main__":
    unittest.main()
