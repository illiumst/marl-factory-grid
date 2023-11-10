import configparser
import json
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':

    conf_path = Path('wg0')
    wg0_conf = configparser.ConfigParser()
    wg0_conf.read(conf_path/'wg0.conf')
    interface = wg0_conf['Interface']
    # Iterate all pears
    for client_name in wg0_conf.sections():
        if client_name == 'Interface':
            continue
        # Delete any old conf.json for the current peer
        (conf_path / f'{client_name}.json').unlink(missing_ok=True)

        peer = wg0_conf[client_name]

        date_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f000Z')

        jdict = dict(
            id=client_name,
            private_key=peer['PublicKey'],
            public_key=peer['PublicKey'],
            # preshared_key=wg0_conf[client_name_wg0]['PresharedKey'],
            name=client_name,
            email=f"sysadmin@mobile.ifi.lmu.de",
            allocated_ips=[interface['Address'].replace('/24', '')],
            allowed_ips=['10.4.0.0/24', '10.153.199.0/24'],
            extra_allowed_ips=[],
            use_server_dns=True,
            enabled=True,
            created_at=date_time,
            updated_at=date_time
        )

        with (conf_path / f'{client_name}.json').open('w+') as f:
            json.dump(jdict, f, indent='\t', separators=(',', ': '))
        print(client_name, ' written...')
