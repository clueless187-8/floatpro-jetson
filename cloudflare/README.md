# FloatPro × Cloudflare Tunnel

Expose the Jetson FastAPI server at a public Cloudflare hostname without
port forwarding, dynamic DNS, or exposing the Jetson to the raw internet.
Optionally gate it with Zero Trust Access so only specific Google / email
accounts can reach the dashboard.

## Why Cloudflare (when you already have Tailscale)

Tailscale is perfect for personal use — your devices, your account, your
mesh. Cloudflare Tunnel is better when:

- **Coaches / parents / players need access** and you don't want to
  install Tailscale on everyone's phone
- **You want a public HTTPS hostname** like `floatpro.tuckersports.com`
  that anyone can bookmark
- **Zero Trust Access gating** — SSO login via Google / GitHub / email
  one-time-pin without writing auth code yourself
- **DDoS protection + CDN** for the static dashboard assets

You can run BOTH. Tailscale for personal access to SSH / RustDesk etc,
Cloudflare Tunnel just for the FloatPro HTTP server.

## Architecture

```
 Browser (coach/player)                Jetson Orin Nano
    │                                        │
    │   HTTPS                                │
    ▼                                        │
 floatpro.yourdomain.com                     │
    │                                        │
    │  (Cloudflare edge)                     │
    │  Zero Trust Access check               │
    │                                        │
    ▼                                        │
 cfargotunnel.com  ────outbound WSS────▶  cloudflared (on Jetson)
                                             │
                                             ▼
                                        uvicorn :8080
                                        (FloatPro FastAPI)
```

No inbound ports on the Jetson. The tunnel is an outbound WebSocket
connection initiated by `cloudflared` from the Jetson to Cloudflare.

## Setup (one-time, ~15 minutes)

You need:

- A domain managed by Cloudflare (free plan is fine)
- `cloudflared` installed on the Jetson

### 1. Install cloudflared on the Jetson (ARM64)

```bash
# Latest .deb for ARM64
curl -L -o cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

### 2. Authenticate to your Cloudflare account

```bash
cloudflared tunnel login
```

This opens a URL. Open it on any device logged into your Cloudflare
dashboard and authorize the zone you want to use. It drops a cert at
`~/.cloudflared/cert.pem`.

### 3. Create a named tunnel

```bash
cloudflared tunnel create floatpro-jetson
```

Note the UUID it prints — you'll need it. It also writes
`~/.cloudflared/<UUID>.json` with the tunnel's credentials.

### 4. Route a public hostname to the tunnel

Pick a hostname under your Cloudflare-managed zone. Example:
`floatpro.tuckersports.com`.

```bash
cloudflared tunnel route dns floatpro-jetson floatpro.tuckersports.com
```

This creates a CNAME record pointing at `<UUID>.cfargotunnel.com` under
the hood. You don't need to touch DNS manually.

### 5. Write the config file

Copy the template in this directory:

```bash
mkdir -p ~/.cloudflared
cp cloudflare/config.yml.template ~/.cloudflared/config.yml
# Edit in <TUNNEL-UUID>, <USER>, <YOUR-HOSTNAME>
nano ~/.cloudflared/config.yml
```

Validate:

```bash
cloudflared tunnel ingress validate
```

### 6. Test it manually first

Start the FastAPI server in one terminal:

```bash
python3 -m floatpro.server --port 8080
```

In another terminal, start the tunnel in the foreground:

```bash
cloudflared tunnel run floatpro-jetson
```

Hit `https://<YOUR-HOSTNAME>` in a browser. You should see the dashboard.

### 7. Install as a systemd service

Once you've confirmed it works manually:

```bash
sudo cloudflared --config ~/.cloudflared/config.yml service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
sudo systemctl status cloudflared
```

The tunnel now comes up automatically whenever the Jetson boots.

For the FastAPI server itself, see `cloudflare/floatpro-server.service`
for a matching systemd unit.

## Gating access with Zero Trust (recommended)

By default the tunnel is public. Anyone with the hostname can hit the
dashboard. For a coach tool that's probably fine for a while, but when
you want to lock it down:

1. In the Cloudflare dashboard, go to **Zero Trust → Access → Applications**
2. Add an application:
   - Type: **Self-hosted**
   - Name: `FloatPro Dashboard`
   - Session duration: 24 hours (or to taste)
   - Application domain: `floatpro.tuckersports.com`
3. Add a policy:
   - Name: `Coaches`
   - Action: `Allow`
   - Include: `Emails ending in @bloomfield-schools.k12.in.us`
     (or specific email list, or GitHub org, etc.)

Now hitting the hostname redirects to a Cloudflare login screen, users
authenticate with Google / email-OTP / whatever, and only then reach the
dashboard. No auth code in your FastAPI server.

## Troubleshooting

**`ERR connection refused`** — tunnel is up but the FastAPI server isn't
running. Check `systemctl status floatpro-server`.

**`websocket: bad handshake`** — usually means the tunnel uuid in
`config.yml` doesn't match the credentials file. Run `cloudflared tunnel
list` and double-check.

**`origin response is taking too long`** — the `/analyze` endpoint runs
spin estimation on all frames, which takes 5-10s on Orin Nano for a
600-frame session. If you see 524 timeouts from Cloudflare, bump
`connectTimeout` in the config or move analysis to a background task
with a polling endpoint.

**Tunnel works over IPv4 but not IPv6** — Cloudflare Tunnel handles
this; if you see issues, set `noHappyEyeballs: true` in the
`originRequest` block (already in the template).

## Further reading

- [Cloudflare Tunnel docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
- [Zero Trust Access policies](https://developers.cloudflare.com/cloudflare-one/policies/access/)
- [Config file reference](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/configure-tunnels/local-management/configuration-file/)
