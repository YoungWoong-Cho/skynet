import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from tests.test_database import DatabaseTestCase


class OperationLockWaitTest(DatabaseTestCase):
    def test_default_blocking_wait_does_not_use_sql_row_lock_deadline(self):
        first = self.database.operation_lock("submission-wait-regression")
        second = self.database.operation_lock("submission-wait-regression")
        first.acquire()
        original = self.database.backend.connect

        def short_sql_timeout():
            connection = original()
            connection.execute("SET lock_timeout='30ms'")
            return connection

        def acquire_then_release():
            with second:
                return True

        with patch.object(self.database.backend, "connect", short_sql_timeout):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(acquire_then_release)
                try:
                    time.sleep(.15)
                    self.assertFalse(future.done())
                finally:
                    first.release()
                self.assertTrue(future.result(timeout=5))
