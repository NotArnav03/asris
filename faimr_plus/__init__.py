"""FAIMR Plus -- optional plugins that extend the core FAIMR audit
framework with heavyweight models the maintainers prefer to keep
out of the core install.

Plugins under this namespace are intentionally lazy-imported and
have their own dependency lists.  Importing faimr_plus itself is a
no-op so the core install does not pay for any of them.
"""
