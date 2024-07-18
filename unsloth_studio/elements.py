# Unsloth Studio
# Copyright (C) 2024-present the Unsloth AI team. All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import uuid

class Click_Button:
    def __init__(self, element, text, callback):
        self.text = text
        self.callback = callback
        self.element = element
    pass

    def _repr_html_(self):
        callback_id = "button-" + str(uuid.uuid4())
        output.register_callback(callback_id, self.callback)
        html = self.element.format(title = self.text, callback_id = callback_id)
        return html
    pass
pass
