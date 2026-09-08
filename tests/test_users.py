# -*- coding: utf-8 -*-
import os, tempfile, unittest
from users import Accounts

SAMPLE = """
users:
  - username: boss
    password: boss123
    role: admin
  - username: ann1
    password: x
"""

class TestAccounts(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.path = os.path.join(d, "users.yaml")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        self.acc = Accounts(self.path)

    def test_authenticate_ok_and_bad(self):
        self.assertTrue(self.acc.authenticate("boss", "boss123"))
        self.assertFalse(self.acc.authenticate("boss", "wrong"))
        self.assertFalse(self.acc.authenticate("nobody", "x"))

    def test_role_and_default_annotator(self):
        self.assertEqual(self.acc.role("boss"), "admin")
        self.assertEqual(self.acc.role("ann1"), "annotator")

    def test_hot_reload_after_edit(self):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("  - username: ann2\n    password: y\n")
        self.assertTrue(self.acc.authenticate("ann2", "y"))
        self.assertEqual(self.acc.role("ann2"), "annotator")

if __name__ == "__main__":
    unittest.main()
