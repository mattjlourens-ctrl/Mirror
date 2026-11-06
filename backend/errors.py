"""Failures the API reports to the browser as {"detail": {"code", "message"}}."""


class ServiceError(Exception):
    """An expected failure; `message` is written for the person using the site."""

    status = 400
    code = "bad_request"

    def __init__(self, message, code=None, status=None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status:
            self.status = status


class InvalidImage(ServiceError):
    status = 415
    code = "unsupported_image"


class FetchFailed(ServiceError):
    """A product page or image couldn't be fetched safely."""

    status = 422
    code = "fetch_failed"


class BlockedDestination(FetchFailed):
    """The URL points at, or redirects to, a private or otherwise disallowed address."""

    code = "blocked_destination"


class AccessRestricted(FetchFailed):
    """The site's robots.txt, or the site itself, doesn't allow automatic access."""

    code = "access_restricted"


class NoProductImages(ServiceError):
    status = 422
    code = "no_product_images"


class Busy(ServiceError):
    status = 503
    code = "busy"


class NotFound(ServiceError):
    status = 404
    code = "not_found"
