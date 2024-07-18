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

finetuning_button = \
"""
<button style="background-color: #008CBA; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; transition-duration: 0.4s; cursor: pointer; border-radius: 12px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);" id="{callback_id}">
    <p style="font-size: 30px;"> {title}</p>
</button>
<script>
    document.querySelector("#{callback_id}").addEventListener('mouseover', function() {{
        this.style.backgroundColor = '#005f73';
        this.style.boxShadow = '0 8px 16px 0 rgba(0,0,0,0.3)';
    }});
    document.querySelector("#{callback_id}").addEventListener('mouseout', function() {{
        this.style.backgroundColor = '#008CBA';
        this.style.boxShadow = '0 4px 8px 0 rgba(0,0,0,0.2)';
    }});
    document.querySelector("#{callback_id}").addEventListener('mousedown', function() {{
        this.style.transform = 'scale(0.95)';
    }});
    document.querySelector("#{callback_id}").addEventListener('mouseup', function() {{
        this.style.transform = 'scale(1)';
    }});
    document.querySelector("#{callback_id}").onclick = (e) => {{
        google.colab.kernel.invokeFunction('{callback_id}', [], {{}})
        e.preventDefault();
    }};
</script>
"""

cycling_emoji = \
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rotating Emoji</title>
    <style>
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .emoji {
            font-size: 25px;
            animation: rotate __SECONDS__ linear;
            animation-duration: __SECONDS__;
        }
    </style>
</head>
<body style="display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background-color: #ffffff; font-family: Arial, sans-serif;">

    <div style="display: flex; justify-content: left; align-items: center; width: 500px; height: 50px;">
        <div class="emoji">
            🦥
        </div>
        <div style="margin-left: 20px;">
            <p style="font-size: 20px;">__TEXT__</p>
        </div>
    </div>

</body>
</html>
"""

