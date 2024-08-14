import launch
from importlib import metadata
from pathlib import Path
from typing import Optional
from packaging.version import parse


repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"


def get_installed_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except Exception:
        return None


def extract_base_package(package_string: str) -> str:
    base_package = package_string.split("@git")[0]
    return base_package


def install_requirements(req_file):
    with open(req_file) as file:
        for package in file:
            try:
                package = package.strip()
                if "==" in package:
                    package_name, package_version = package.split("==")
                    installed_version = get_installed_version(package_name)
                    if installed_version != package_version:
                        launch.run_pip(
                            f'install -U "{package}"',
                            f"sd-webui-birefnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif ">=" in package:
                    package_name, package_version = package.split(">=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or parse(
                        installed_version
                    ) < parse(package_version):
                        launch.run_pip(
                            f'install -U "{package}"',
                            f"sd-webui-birefnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif "<=" in package:
                    package_name, package_version = package.split("<=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or parse(
                        installed_version
                    ) > parse(package_version):
                        launch.run_pip(
                            f'install "{package_name}=={package_version}"',
                            f"sd-webui-birefnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif "<" in package:
                    package_name, package_version = package.split("<")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or parse(
                        installed_version
                    ) >= parse(package_version):
                        launch.run_pip(
                            f'install "{package_name}=={package_version}"',
                            f"sd-webui-birefnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif not launch.is_installed(extract_base_package(package)):
                    launch.run_pip(
                        f'install "{package}"',
                        f"sd-webui-birefnet requirement: {package}",
                    )
            except Exception as e:
                print(e)
                print(
                    f"Warning: Failed to install {package}, some preprocessors may not work."
                )


install_requirements(main_req_file)